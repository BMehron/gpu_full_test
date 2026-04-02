#!/usr/bin/env python3
"""
Per-GPU tests — run independently on a single GPU.

Usage (direct):
    python tests/per_gpu.py --gpu-id 0 --config '{"memory_fraction":0.5}' --output /tmp/gpu0.json

The orchestrator launches one process per GPU sequentially.
"""

import argparse
import json
import sys
import time
import traceback
from typing import Dict, List

import torch
import torch.nn as nn


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def result(name: str, status: str, metrics: Dict = None, details: str = "", error: str = "") -> Dict:
    return {"name": name, "status": status, "metrics": metrics or {}, "details": details, "error": error}


def cuda_time_ms(fn, iters: int, device) -> float:
    """Return average kernel time in ms using wall-clock time after GPU sync."""
    for _ in range(max(3, iters // 10)):
        fn()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters * 1000


def save_loss_plot(losses: List[float], title: str, output_json_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plot_path = output_json_path.replace(".json", "_loss.png")
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Loss plot saved: {plot_path}", flush=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Test: GPU info
# ---------------------------------------------------------------------------

def test_gpu_info(gpu_id: int, cfg: Dict) -> Dict:
    """Collect device properties — always PASS, used as header in report."""
    try:
        p = torch.cuda.get_device_properties(gpu_id)
        mem_gb = p.total_memory / 1024**3
        metrics = {
            "name": p.name,
            "total_memory_gb": round(mem_gb, 2),
            "sm_count": p.multi_processor_count,
            "compute_capability": f"{p.major}.{p.minor}",
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        }
        return result("gpu_info", PASS, metrics,
                      details=f"{p.name} | {mem_gb:.1f} GB VRAM | CC {p.major}.{p.minor} | CUDA {torch.version.cuda}")
    except Exception:
        return result("gpu_info", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: MatMul correctness
# ---------------------------------------------------------------------------

def test_matmul_correctness(gpu_id: int, cfg: Dict) -> Dict:
    """
    BF16 matmul on GPU vs FP32 on CPU.
    Checks Frobenius relative error against configurable threshold.
    """
    try:
        size      = cfg.get("matmul_size", 8192)
        threshold = cfg.get("matmul_err_threshold", 0.02)
        seed      = cfg.get("training_seed", 42)
        device    = torch.device(f"cuda:{gpu_id}")
        torch.manual_seed(seed)

        ref_size = 1024
        A_f = torch.randn(ref_size, ref_size)
        B_f = torch.randn(ref_size, ref_size)
        ref = A_f @ B_f

        A_g = A_f.to(device=device, dtype=torch.bfloat16)
        B_g = B_f.to(device=device, dtype=torch.bfloat16)
        out = (A_g @ B_g).float().cpu()

        rel_err = (out - ref).norm().item() / (ref.norm().item() + 1e-8)
        status = PASS if rel_err < threshold else FAIL

        A_l = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        B_l = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        ms = cuda_time_ms(lambda: torch.mm(A_l, B_l), iters=20, device=device)
        tflops = 2 * size**3 / (ms / 1000) / 1e12

        del A_f, B_f, ref, A_g, B_g, out, A_l, B_l
        torch.cuda.empty_cache()

        return result("matmul_correctness", status,
                      metrics={"frobenius_rel_err": round(rel_err, 6), "threshold": threshold,
                                "large_matmul_tflops_bf16": round(tflops, 1), "matrix_size": size},
                      details=f"frobenius_rel_err={rel_err:.4%} (thr {threshold:.0%}) | "
                              f"large matmul {tflops:.1f} TFLOPS BF16")
    except Exception:
        return result("matmul_correctness", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: Memory allocation + pattern verification
# ---------------------------------------------------------------------------

def test_memory(gpu_id: int, cfg: Dict) -> Dict:
    """Allocate fraction of VRAM, fill with known pattern, verify no corruption."""
    try:
        fraction = cfg.get("memory_fraction", 0.5)
        device   = torch.device(f"cuda:{gpu_id}")
        props    = torch.cuda.get_device_properties(gpu_id)
        total_bytes = props.total_memory
        alloc_bytes = int(total_bytes * fraction)
        n_elems     = alloc_bytes // 4

        t0 = time.perf_counter()
        t  = torch.empty(n_elems, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        alloc_s = time.perf_counter() - t0

        pattern  = 3.14159265
        t.fill_(pattern)
        torch.cuda.synchronize(device)
        readback = t.mean().item()
        correct  = abs(readback - pattern) < 1e-4

        del t
        torch.cuda.empty_cache()

        alloc_gb = alloc_bytes / 1024**3
        status   = PASS if correct else FAIL
        return result("memory", status,
                      metrics={"total_vram_gb": round(total_bytes / 1024**3, 2),
                                "allocated_gb": round(alloc_gb, 2), "fraction": fraction,
                                "alloc_time_s": round(alloc_s, 3), "pattern_ok": correct},
                      details=f"Allocated {alloc_gb:.1f} GB ({fraction*100:.0f}% VRAM) in {alloc_s:.2f}s | "
                              f"pattern {'OK' if correct else 'CORRUPTED'}")
    except torch.cuda.OutOfMemoryError as e:
        return result("memory", FAIL, error=f"OOM: {e}")
    except Exception:
        return result("memory", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: PCIe Host ↔ Device bandwidth
# ---------------------------------------------------------------------------

def test_pcie_bandwidth(gpu_id: int, cfg: Dict) -> Dict:
    """Pinned-memory H2D and D2H bandwidth."""
    try:
        data_gb   = cfg.get("bandwidth_data_gb", 2.0)
        threshold = cfg.get("pcie_threshold_gbps", 25.0)
        device    = torch.device(f"cuda:{gpu_id}")
        n         = int(data_gb * 1024**3) // 4
        iters     = 5

        cpu_t = torch.zeros(n, dtype=torch.float32).pin_memory()
        gpu_t = torch.zeros(n, dtype=torch.float32, device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            gpu_t.copy_(cpu_t, non_blocking=False)
            torch.cuda.synchronize(device)
        h2d = data_gb * iters / (time.perf_counter() - t0)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            cpu_t.copy_(gpu_t, non_blocking=False)
            torch.cuda.synchronize(device)
        d2h = data_gb * iters / (time.perf_counter() - t0)

        del cpu_t, gpu_t
        torch.cuda.empty_cache()

        status = PASS if (h2d >= threshold and d2h >= threshold) else WARN
        return result("pcie_bandwidth", status,
                      metrics={"h2d_gbps": round(h2d, 2), "d2h_gbps": round(d2h, 2),
                                "threshold_gbps": threshold, "data_gb": data_gb},
                      details=f"H2D {h2d:.1f} GB/s | D2H {d2h:.1f} GB/s | threshold {threshold} GB/s")
    except Exception:
        return result("pcie_bandwidth", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: HBM bandwidth
# ---------------------------------------------------------------------------

def test_hbm_bandwidth(gpu_id: int, cfg: Dict) -> Dict:
    """Device memcpy bandwidth — proxy for HBM throughput."""
    try:
        data_gb   = cfg.get("hbm_data_gb", 4.0)
        threshold = cfg.get("hbm_threshold_tbps", 2.5)
        device    = torch.device(f"cuda:{gpu_id}")
        n         = int(data_gb * 1024**3) // 4

        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)

        ms     = cuda_time_ms(lambda: dst.copy_(src), iters=10, device=device)
        bw_tbps = (2 * data_gb) / (ms / 1000) / 1000

        del src, dst
        torch.cuda.empty_cache()

        status = PASS if bw_tbps >= threshold else WARN
        return result("hbm_bandwidth", status,
                      metrics={"bandwidth_tbps": round(bw_tbps, 3),
                                "bandwidth_gbps": round(bw_tbps * 1000, 1),
                                "threshold_tbps": threshold},
                      details=f"{bw_tbps:.3f} TB/s ({bw_tbps*1000:.0f} GB/s) | threshold {threshold} TB/s")
    except Exception:
        return result("hbm_bandwidth", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: BF16 TFLOPS
# ---------------------------------------------------------------------------

def test_tflops_bf16(gpu_id: int, cfg: Dict) -> Dict:
    """Sustained BF16 Tensor Core throughput."""
    try:
        iters       = cfg.get("tflops_iters", 200)
        size        = cfg.get("matmul_size", 8192)
        peak_tflops = cfg.get("h200_peak_bf16_tflops", 989.0)
        threshold   = cfg.get("tflops_threshold_pct", 65.0)
        device      = torch.device(f"cuda:{gpu_id}")

        A = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        B = torch.randn(size, size, dtype=torch.bfloat16, device=device)

        ms       = cuda_time_ms(lambda: torch.mm(A, B), iters=iters, device=device)
        achieved = 2 * size**3 / (ms / 1000) / 1e12
        util     = achieved / peak_tflops * 100

        del A, B
        torch.cuda.empty_cache()

        status = PASS if util >= threshold else WARN
        return result("tflops_bf16", status,
                      metrics={"achieved_tflops": round(achieved, 1), "peak_tflops": peak_tflops,
                                "utilization_pct": round(util, 1), "threshold_pct": threshold,
                                "matrix_size": size},
                      details=f"{achieved:.1f} TFLOPS BF16 ({util:.1f}% of {peak_tflops:.0f} peak)")
    except Exception:
        return result("tflops_bf16", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: Mixed-precision correctness
# ---------------------------------------------------------------------------

def test_mixed_precision_correctness(gpu_id: int, cfg: Dict) -> Dict:
    """BF16 inputs with FP32 accumulation vs FP64 reference."""
    try:
        threshold = cfg.get("mixed_precision_err_threshold", 0.005)
        seed      = cfg.get("training_seed", 42)
        device    = torch.device(f"cuda:{gpu_id}")
        torch.manual_seed(seed)
        size = 512

        A = torch.randn(size, size)
        B = torch.randn(size, size)
        ref = (A.double() @ B.double()).float()

        A_g = A.to(device=device, dtype=torch.bfloat16)
        B_g = B.to(device=device, dtype=torch.bfloat16)
        out = torch.mm(A_g.float(), B_g.float()).cpu()

        rel_err = (out - ref).norm().item() / (ref.norm().item() + 1e-8)
        status  = PASS if rel_err < threshold else FAIL
        return result("mixed_precision_correctness", status,
                      metrics={"frobenius_rel_err_fp32_vs_fp64": round(rel_err, 8), "threshold": threshold},
                      details=f"FP32 accum vs FP64 ref: frobenius_rel_err={rel_err:.6%}")
    except Exception:
        return result("mixed_precision_correctness", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: Single-GPU training loop
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_training(gpu_id: int, cfg: Dict) -> Dict:
    """
    Single-GPU training on y = X @ w (linear target, guaranteed convergence).
    Fixed seed for reproducibility; each GPU trains independently on the same data.
    """
    try:
        hidden     = cfg.get("ddp_hidden_size", 1024)
        steps      = cfg.get("ddp_steps", 50)
        batch_size = cfg.get("ddp_batch_size", 64)
        threshold  = cfg.get("ddp_loss_drop_threshold", 0.5)
        seed       = cfg.get("training_seed", 42)
        device     = torch.device(f"cuda:{gpu_id}")

        torch.manual_seed(seed)
        model     = SimpleMLP(hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        torch.manual_seed(seed)
        w = torch.randn(hidden, 1, device=device)
        X = torch.randn(batch_size, hidden, device=device)
        y = X @ w

        losses = []
        t0 = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(round(loss.item(), 6))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        del model, optimizer, X, y, w
        torch.cuda.empty_cache()

        first_loss, last_loss = losses[0], losses[-1]
        relative_drop = (first_loss - last_loss) / (first_loss + 1e-8)
        throughput    = batch_size * steps / elapsed
        status        = PASS if relative_drop >= threshold else FAIL
        return result("training", status,
                      metrics={"first_loss": first_loss, "last_loss": last_loss,
                                "relative_drop_pct": round(relative_drop * 100, 1),
                                "threshold_pct": threshold * 100,
                                "throughput_samples_per_s": round(throughput, 1),
                                "steps": steps, "step_losses": losses},
                      details=f"loss {first_loss:.4f}→{last_loss:.4f} "
                              f"({relative_drop*100:.1f}% drop) | {throughput:.0f} samples/s")
    except Exception:
        return result("training", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(gpu_id: int, cfg: Dict) -> Dict:
    device = torch.device(f"cuda:{gpu_id}")
    _ = torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize(device)

    tests = [
        lambda: test_gpu_info(gpu_id, cfg),
        lambda: test_matmul_correctness(gpu_id, cfg),
        lambda: test_memory(gpu_id, cfg),
        lambda: test_pcie_bandwidth(gpu_id, cfg),
        lambda: test_hbm_bandwidth(gpu_id, cfg),
        lambda: test_tflops_bf16(gpu_id, cfg),
        lambda: test_mixed_precision_correctness(gpu_id, cfg),
        lambda: test_training(gpu_id, cfg),
    ]

    sym = {PASS: "✓", FAIL: "✗", WARN: "△"}
    results = []
    for fn in tests:
        r = fn()
        results.append(r)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        tag = sym.get(r["status"], "?")
        msg = r["details"] if r["details"] else r["error"]
        print(f"  GPU{gpu_id} [{tag}] {r['name']:<35} {msg}", flush=True)

    passed = sum(1 for r in results if r["status"] == PASS)
    failed = sum(1 for r in results if r["status"] == FAIL)
    warned = sum(1 for r in results if r["status"] == WARN)
    return {
        "gpu_id": gpu_id,
        "tests": results,
        "summary": {"total": len(results), "passed": passed, "failed": failed, "warned": warned},
    }


def main():
    parser = argparse.ArgumentParser(description="Per-GPU tests")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--config", type=str, default="{}")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)
    if args.gpu_id >= torch.cuda.device_count():
        print(f"ERROR: GPU {args.gpu_id} not found ({torch.cuda.device_count()} GPUs present)",
              file=sys.stderr)
        sys.exit(1)

    cfg     = json.loads(args.config)
    results = run_all(args.gpu_id, cfg)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    for t in results["tests"]:
        losses = t.get("metrics", {}).get("step_losses")
        if losses:
            save_loss_plot(losses, f"GPU{args.gpu_id} {t['name']}", args.output)


if __name__ == "__main__":
    main()
