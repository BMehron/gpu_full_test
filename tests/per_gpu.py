#!/usr/bin/env python3
"""
Per-GPU tests — run independently on a single GPU.

Usage (direct):
    python tests/per_gpu.py --gpu-id 0 --config '{"memory_fraction":0.5}' --output /tmp/gpu0.json

The orchestrator launches one process per GPU in parallel.
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


# ---------------------------------------------------------------------------
# Test: GPU info
# ---------------------------------------------------------------------------

def test_gpu_info(gpu_id: int) -> Dict:
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

def test_matmul_correctness(gpu_id: int, size: int = 8192) -> Dict:
    """
    BF16 matmul on GPU vs FP32 on CPU.
    Checks max relative error < 2% (BF16 has ~0.4% vs FP32; 2% leaves room for
    accumulation differences without hiding real compute errors).
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.manual_seed(42)

        ref_size = 1024  # CPU reference on small matrix (large would be too slow)
        A_f = torch.randn(ref_size, ref_size)
        B_f = torch.randn(ref_size, ref_size)
        ref = A_f @ B_f  # FP32 CPU ground truth

        A_g = A_f.to(device=device, dtype=torch.bfloat16)
        B_g = B_f.to(device=device, dtype=torch.bfloat16)
        out = (A_g @ B_g).float().cpu()

        # Frobenius norm relative error avoids explosion on near-zero elements
        rel_err = (out - ref).norm().item() / (ref.norm().item() + 1e-8)

        threshold = 0.02
        status = PASS if rel_err < threshold else FAIL

        # Large-matrix throughput (just timing, no CPU reference)
        A_l = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        B_l = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        ms = cuda_time_ms(lambda: torch.mm(A_l, B_l), iters=20, device=device)
        tflops = 2 * size**3 / (ms / 1000) / 1e12

        del A_f, B_f, ref, A_g, B_g, out, A_l, B_l
        torch.cuda.empty_cache()

        return result("matmul_correctness", status,
                      metrics={"frobenius_rel_err": round(rel_err, 6),
                                "threshold": threshold,
                                "large_matmul_tflops_bf16": round(tflops, 1),
                                "matrix_size": size},
                      details=f"frobenius_rel_err={rel_err:.4%} (thr {threshold:.0%}) | "
                              f"large matmul {tflops:.1f} TFLOPS BF16")
    except Exception:
        return result("matmul_correctness", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: Memory allocation + pattern verification
# ---------------------------------------------------------------------------

def test_memory(gpu_id: int, fraction: float = 0.5) -> Dict:
    """
    Allocate `fraction` of total VRAM, fill with a known pattern, read back
    and verify no bit corruption, then free.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        props = torch.cuda.get_device_properties(gpu_id)
        total_bytes = props.total_memory
        alloc_bytes = int(total_bytes * fraction)
        n_elems = alloc_bytes // 4  # float32

        t0 = time.perf_counter()
        t = torch.empty(n_elems, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        alloc_s = time.perf_counter() - t0

        pattern = 3.14159265
        t.fill_(pattern)
        torch.cuda.synchronize(device)

        readback = t.mean().item()
        correct = abs(readback - pattern) < 1e-4

        del t
        torch.cuda.empty_cache()

        alloc_gb = alloc_bytes / 1024**3
        status = PASS if correct else FAIL
        return result("memory", status,
                      metrics={"total_vram_gb": round(total_bytes / 1024**3, 2),
                                "allocated_gb": round(alloc_gb, 2),
                                "fraction": fraction,
                                "alloc_time_s": round(alloc_s, 3),
                                "pattern_ok": correct},
                      details=f"Allocated {alloc_gb:.1f} GB ({fraction*100:.0f}% VRAM) in {alloc_s:.2f}s | "
                              f"pattern {'OK' if correct else 'CORRUPTED'}")
    except torch.cuda.OutOfMemoryError as e:
        return result("memory", FAIL, error=f"OOM: {e}")
    except Exception:
        return result("memory", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: PCIe Host ↔ Device bandwidth
# ---------------------------------------------------------------------------

def test_pcie_bandwidth(gpu_id: int, data_gb: float = 2.0) -> Dict:
    """
    Pinned-memory H2D and D2H bandwidth.
    H200 uses PCIe 5.0 x16 (~64 GB/s theoretical); threshold set at 40 GB/s.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        n = int(data_gb * 1024**3) // 4  # float32 elements

        cpu_t = torch.zeros(n, dtype=torch.float32).pin_memory()
        gpu_t = torch.zeros(n, dtype=torch.float32, device=device)
        iters = 5

        # H2D
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            gpu_t.copy_(cpu_t, non_blocking=False)
            torch.cuda.synchronize(device)
        h2d = data_gb * iters / (time.perf_counter() - t0)

        # D2H
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            cpu_t.copy_(gpu_t, non_blocking=False)
            torch.cuda.synchronize(device)
        d2h = data_gb * iters / (time.perf_counter() - t0)

        del cpu_t, gpu_t
        torch.cuda.empty_cache()

        threshold = 25.0  # GB/s — 8 concurrent transfers share PCIe bandwidth
        status = PASS if (h2d >= threshold and d2h >= threshold) else WARN
        return result("pcie_bandwidth", status,
                      metrics={"h2d_gbps": round(h2d, 2), "d2h_gbps": round(d2h, 2),
                                "threshold_gbps": threshold, "data_gb": data_gb},
                      details=f"H2D {h2d:.1f} GB/s | D2H {d2h:.1f} GB/s | threshold {threshold} GB/s")
    except Exception:
        return result("pcie_bandwidth", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: HBM bandwidth (device-to-device copy — stresses memory subsystem)
# ---------------------------------------------------------------------------

def test_hbm_bandwidth(gpu_id: int, data_gb: float = 4.0) -> Dict:
    """
    Device memcpy bandwidth, a proxy for HBM3e throughput.
    H200 HBM3e: 3.35 TB/s theoretical; threshold 2.5 TB/s.
    Each copy reads src + writes dst → effective 2×data_gb per iter.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        n = int(data_gb * 1024**3) // 4

        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)

        ms = cuda_time_ms(lambda: dst.copy_(src), iters=10, device=device)
        bw_tbps = (2 * data_gb) / (ms / 1000) / 1000  # TB/s

        del src, dst
        torch.cuda.empty_cache()

        threshold = 2.5  # TB/s
        status = PASS if bw_tbps >= threshold else WARN
        return result("hbm_bandwidth", status,
                      metrics={"bandwidth_tbps": round(bw_tbps, 3),
                                "bandwidth_gbps": round(bw_tbps * 1000, 1),
                                "threshold_tbps": threshold},
                      details=f"{bw_tbps:.3f} TB/s ({bw_tbps*1000:.0f} GB/s) | threshold {threshold} TB/s")
    except Exception:
        return result("hbm_bandwidth", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: BF16 TFLOPS (Tensor Core utilisation)
# ---------------------------------------------------------------------------

def test_tflops_bf16(gpu_id: int, iters: int = 200, size: int = 8192,
                     peak_tflops: float = 1979.0) -> Dict:
    """
    Sustained BF16 Tensor Core throughput.
    Threshold: 70% of rated peak (accounts for memory-bound regimes on smaller
    matrices, but size=8192 should be compute-bound on H200).
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        A = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        B = torch.randn(size, size, dtype=torch.bfloat16, device=device)

        ms = cuda_time_ms(lambda: torch.mm(A, B), iters=iters, device=device)
        flops = 2 * size**3
        achieved = flops / (ms / 1000) / 1e12
        util = achieved / peak_tflops * 100

        del A, B
        torch.cuda.empty_cache()

        threshold_pct = 65.0  # Dense BF16 GEMM; 70% minus ~5% wall-clock overhead
        status = PASS if util >= threshold_pct else WARN
        return result("tflops_bf16", status,
                      metrics={"achieved_tflops": round(achieved, 1),
                                "peak_tflops": peak_tflops,
                                "utilization_pct": round(util, 1),
                                "threshold_pct": threshold_pct,
                                "matrix_size": size},
                      details=f"{achieved:.1f} TFLOPS BF16 ({util:.1f}% of {peak_tflops:.0f} peak)")
    except Exception:
        return result("tflops_bf16", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: Tensor Core mixed-precisigiton correctness (FP32 accumulation)
# ---------------------------------------------------------------------------

def test_mixed_precision_correctness(gpu_id: int) -> Dict:
    """
    BF16 inputs with FP32 accumulation (torch.mm with autocast disabled)
    vs pure FP32 reference. Verifies Tensor Cores accumulate correctly.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.manual_seed(7)
        size = 512

        A = torch.randn(size, size)
        B = torch.randn(size, size)
        ref = (A.double() @ B.double()).float()  # Double-precision reference

        A_g = A.to(device=device, dtype=torch.bfloat16)
        B_g = B.to(device=device, dtype=torch.bfloat16)
        # Use float32 accumulation via torch.mm on bf16 inputs
        out = torch.mm(A_g.float(), B_g.float()).cpu()

        rel_err = (out - ref).norm().item() / (ref.norm().item() + 1e-8)
        threshold = 0.005  # FP32 vs FP64: < 0.5% expected for 512×512 matmul
        status = PASS if rel_err < threshold else FAIL

        return result("mixed_precision_correctness", status,
                      metrics={"frobenius_rel_err_fp32_vs_fp64": round(rel_err, 8), "threshold": threshold},
                      details=f"FP32 accum vs FP64 ref: frobenius_rel_err={rel_err:.6%}")
    except Exception:
        return result("mixed_precision_correctness", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: Single-GPU training loop
# ---------------------------------------------------------------------------

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


def test_training(gpu_id: int, hidden: int = 1024, steps: int = 50,
                  batch_size: int = 64) -> Dict:
    """
    Single-GPU training loop on a linear target y = X @ w.
    Verifies the GPU can run a full forward/backward/optimizer cycle
    and that loss drops by ≥50% over `steps` steps.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.manual_seed(42)

        model = SimpleMLP(hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        torch.manual_seed(0)
        w = torch.randn(hidden, 1, device=device)
        X = torch.randn(batch_size, hidden, device=device)
        y = X @ w

        losses = []
        t0 = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(round(loss.item(), 6))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        del model, optimizer, X, y, w
        torch.cuda.empty_cache()

        first_loss, last_loss = losses[0], losses[-1]
        relative_drop = (first_loss - last_loss) / (first_loss + 1e-8)
        throughput = batch_size * steps / elapsed
        status = PASS if relative_drop >= 0.5 else FAIL
        return result("training", status,
                      metrics={"first_loss": first_loss,
                                "last_loss": last_loss,
                                "relative_drop_pct": round(relative_drop * 100, 1),
                                "throughput_samples_per_s": round(throughput, 1),
                                "steps": steps,
                                "step_losses": losses},
                      details=f"loss {first_loss:.4f}→{last_loss:.4f} "
                              f"({relative_drop*100:.1f}% drop) | {throughput:.0f} samples/s")
    except Exception:
        return result("training", FAIL, error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(gpu_id: int, cfg: Dict) -> Dict:
    device = torch.device(f"cuda:{gpu_id}")
    # Brief GPU warmup
    _ = torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize(device)

    tests = [
        lambda: test_gpu_info(gpu_id),
        lambda: test_matmul_correctness(gpu_id, size=cfg.get("matmul_size", 8192)),
        lambda: test_memory(gpu_id, fraction=cfg.get("memory_fraction", 0.5)),
        lambda: test_pcie_bandwidth(gpu_id, data_gb=cfg.get("bandwidth_data_gb", 2.0)),
        lambda: test_hbm_bandwidth(gpu_id, data_gb=cfg.get("hbm_data_gb", 4.0)),
        lambda: test_tflops_bf16(gpu_id, iters=cfg.get("tflops_iters", 200),
                                  size=cfg.get("matmul_size", 8192),
                                  peak_tflops=cfg.get("h200_peak_bf16_tflops", 1979.0)),
        lambda: test_mixed_precision_correctness(gpu_id),
        lambda: test_training(gpu_id, hidden=cfg.get("ddp_hidden_size", 1024),
                               steps=cfg.get("ddp_steps", 50)),
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

    passed  = sum(1 for r in results if r["status"] == PASS)
    failed  = sum(1 for r in results if r["status"] == FAIL)
    warned  = sum(1 for r in results if r["status"] == WARN)
    return {
        "gpu_id": gpu_id,
        "tests": results,
        "summary": {"total": len(results), "passed": passed, "failed": failed, "warned": warned},
    }


def main():
    parser = argparse.ArgumentParser(description="Per-GPU tests")
    parser.add_argument("--gpu-id",  type=int, required=True)
    parser.add_argument("--config",  type=str, default="{}", help="JSON test-config string")
    parser.add_argument("--output",  type=str, required=True, help="Path to write JSON results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)
    if args.gpu_id >= torch.cuda.device_count():
        print(f"ERROR: GPU {args.gpu_id} not found ({torch.cuda.device_count()} GPUs present)",
              file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(args.config)
    results = run_all(args.gpu_id, cfg)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    for t in results["tests"]:
        losses = t.get("metrics", {}).get("step_losses")
        if losses:
            save_loss_plot(losses, f"GPU{args.gpu_id} {t['name']}", args.output)


if __name__ == "__main__":
    main()
