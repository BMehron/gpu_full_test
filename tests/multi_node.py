#!/usr/bin/env python3
"""
Multi-node distributed tests across all nodes.

Launched by the orchestrator on EVERY node simultaneously:
    torchrun --nnodes=N --nproc_per_node=8 \
             --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:<port> \
             --rdzv_id=gpu_test \
             tests/multi_node.py --config '...' --output /tmp/multi_node.json

Global rank 0 writes the result file.
NODE_RANK=0 is the master node (first in config).
"""

import argparse
import json
import os
import time
import traceback
from typing import Dict, List, Optional


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

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def result(name, status, metrics=None, details="", error=""):
    return {"name": name, "status": status, "metrics": metrics or {}, "details": details, "error": error}


def init_dist():
    dist.init_process_group(backend="nccl")
    rank         = dist.get_rank()
    world        = dist.get_world_size()
    local_rank   = int(os.environ.get("LOCAL_RANK", 0))
    node_rank    = int(os.environ.get("NODE_RANK", 0))
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, node_rank, gpus_per_node


def barrier_sync(rank, world):
    dist.barrier()
    torch.cuda.synchronize()


def all_gather_scalar(val, rank, world):
    t = torch.tensor([val], dtype=torch.float64, device=f"cuda:{torch.cuda.current_device()}")
    gathered = [torch.zeros(1, dtype=torch.float64, device=t.device) for _ in range(world)]
    dist.all_gather(gathered, t)
    return [g.item() for g in gathered]


# ---------------------------------------------------------------------------
# Test: Cross-node AllReduce correctness
# ---------------------------------------------------------------------------

def test_allreduce_correctness(rank, world, local_rank):
    """
    Each rank contributes (rank+1). Expected sum = world*(world+1)/2.
    Tested for fp32 and bf16.
    """
    try:
        dev = f"cuda:{local_rank}"
        errors = []
        expected = world * (world + 1) / 2

        for dtype, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
            t = torch.full((1024,), float(rank + 1), dtype=dtype, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            got = t[0].float().item()
            rel_err = abs(got - expected) / expected
            tol = 0.001 if dtype == torch.float32 else 0.01
            if rel_err > tol:
                errors.append(f"{name}: expected {expected:.1f} got {got:.4f} rel_err={rel_err:.2e}")

        if rank != 0:
            return None

        status = PASS if not errors else FAIL
        return result("allreduce_correctness_multi_node", status,
                      metrics={"errors": errors, "world_size": world, "nodes": world // 8},
                      details="fp32 + bf16 cross-node allreduce correct" if not errors else str(errors))
    except Exception:
        if rank == 0:
            return result("allreduce_correctness_multi_node", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: Cross-node AllReduce bandwidth sweep
# ---------------------------------------------------------------------------

def test_allreduce_bandwidth(rank, world, local_rank, sizes_mb=None, cfg=None):
    """
    Measures inter-node AllReduce bus bandwidth at multiple payload sizes.
    Bus BW formula: 2*(N-1)/N * size / time.
    """
    if cfg is None:
        cfg = {}
    if sizes_mb is None:
        sizes_mb = [64, 256, 1024]

    try:
        dev = f"cuda:{local_rank}"
        bw_results = {}

        for size_mb in sizes_mb:
            n = int(size_mb * 1024**2) // 4
            t = torch.ones(n, dtype=torch.float32, device=dev)

            for _ in range(3):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            barrier_sync(rank, world)

            iters = 5
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            barrier_sync(rank, world)
            elapsed = time.perf_counter() - t0

            size_bytes = size_mb * 1024**2
            bus_bw = 2 * (world - 1) / world * size_bytes * iters / elapsed / 1e9
            bw_results[size_mb] = round(bus_bw, 1)
            del t
            torch.cuda.empty_cache()

        if rank != 0:
            return None

        min_bw = min(bw_results.values())
        threshold = cfg.get("allreduce_threshold_gbps", 2.0)
        status = PASS if min_bw >= threshold else WARN
        details = " | ".join(f"{s}MB→{b}GB/s" for s, b in bw_results.items())
        return result("allreduce_bandwidth_multi_node", status,
                      metrics={"by_size_mb_gbps": bw_results, "min_gbps": min_bw,
                                "threshold_gbps": threshold},
                      details=details)
    except Exception:
        if rank == 0:
            return result("allreduce_bandwidth_multi_node", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: Multi-node DDP training
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_ddp_training(rank, world, local_rank, hidden=4096, steps=50, batch_size=64,
                      threshold=0.5, seed=42):
    """
    Multi-node DDP training loop:
    1. Runs `steps` forward+backward+optimizer steps.
    2. Verifies loss drops by ≥threshold (linear target y = X @ w is perfectly learnable).
    3. Checks gradient norms are identical across ALL ranks (cross-node allreduce).
    4. Reports cross-node throughput (samples/s).
    """
    try:
        device = torch.device(f"cuda:{local_rank}")
        torch.manual_seed(seed)

        model     = SimpleMLP(hidden).to(device)
        ddp_model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Fixed seed: generate full dataset then slice per rank → reproducible + different data per rank
        torch.manual_seed(seed)
        w     = torch.randn(hidden, 1, device=device)
        X_all = torch.randn(batch_size * world, hidden, device=device)
        X     = X_all[rank * batch_size:(rank + 1) * batch_size]
        y     = X @ w

        first_loss = None
        last_loss = None
        grad_sync_ok = True
        losses = []

        barrier_sync(rank, world)
        t0 = time.perf_counter()

        for step in range(steps):
            optimizer.zero_grad()
            out = ddp_model(X)
            loss = criterion(out, y)
            loss.backward()

            if step % 10 == 0:
                gn = ddp_model.module.net[0].weight.grad.norm().item()
                all_gns = all_gather_scalar(gn, rank, world)
                if rank == 0:
                    mn, mx = min(all_gns), max(all_gns)
                    if mx > 0 and (mx - mn) / mx > 0.001:
                        grad_sync_ok = False

            optimizer.step()
            lv = loss.item()
            if first_loss is None:
                first_loss = lv
            last_loss = lv
            losses.append(round(lv, 6))

        torch.cuda.synchronize()
        barrier_sync(rank, world)
        elapsed = time.perf_counter() - t0

        throughput = batch_size * steps * world / elapsed

        del ddp_model, model, optimizer, X, y, w
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        relative_drop = (first_loss - last_loss) / (first_loss + 1e-8)
        status = PASS if (relative_drop >= threshold and grad_sync_ok) else FAIL
        return result("ddp_training_multi_node", status,
                      metrics={"first_loss": round(first_loss, 6),
                                "last_loss": round(last_loss, 6),
                                "relative_drop_pct": round(relative_drop * 100, 1),
                                "grad_sync_ok": grad_sync_ok,
                                "throughput_samples_per_s": round(throughput, 1),
                                "world_size": world,
                                "nodes": world // 8,
                                "step_losses": losses},
                      details=f"loss {first_loss:.4f}→{last_loss:.4f} ({relative_drop*100:.1f}% drop) | "
                              f"grad_sync {'OK' if grad_sync_ok else 'FAIL'} | "
                              f"{throughput:.0f} samples/s across {world} GPUs")
    except Exception:
        if rank == 0:
            return result("ddp_training_multi_node", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="{}")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rank, world, local_rank, node_rank, gpus_per_node = init_dist()
    cfg = json.loads(args.config)

    if rank == 0:
        print(f"\n=== Multi-Node Tests (world={world}, nodes={world // gpus_per_node}) ===", flush=True)
        print("  Initializing NCCL …", flush=True)
    _w = torch.zeros(1, device=f"cuda:{local_rank}")
    dist.all_reduce(_w)
    torch.cuda.synchronize()
    del _w
    if rank == 0:
        print("  NCCL ready.", flush=True)

    sym = {PASS: "✓", FAIL: "✗", WARN: "△"}
    all_results = []

    def run(fn):
        r = fn()
        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if r is not None:
            all_results.append(r)
            tag = sym.get(r["status"], "?")
            msg = r["details"] if r["details"] else r["error"]
            print(f"  [{tag}] {r['name']:<45} {msg}", flush=True)
        return r

    run(lambda: test_allreduce_correctness(rank, world, local_rank))
    run(lambda: test_allreduce_bandwidth(rank, world, local_rank,
                                          sizes_mb=cfg.get("allreduce_sizes_mb", [64, 256, 1024]),
                                          cfg=cfg))
    run(lambda: test_ddp_training(rank, world, local_rank,
                                   hidden=cfg.get("ddp_hidden_size", 4096),
                                   steps=cfg.get("ddp_steps", 20),
                                   batch_size=cfg.get("ddp_batch_size", 64),
                                   threshold=cfg.get("ddp_loss_drop_threshold", 0.5),
                                   seed=cfg.get("training_seed", 42)))

    dist.destroy_process_group()

    if rank == 0:
        passed = sum(1 for r in all_results if r["status"] == PASS)
        failed = sum(1 for r in all_results if r["status"] == FAIL)
        warned = sum(1 for r in all_results if r["status"] == WARN)
        output = {
            "type": "multi_node",
            "world_size": world,
            "nodes": world // gpus_per_node,
            "tests": all_results,
            "summary": {"total": len(all_results), "passed": passed, "failed": failed, "warned": warned},
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        for r in all_results:
            losses = r.get("metrics", {}).get("step_losses")
            if losses:
                save_loss_plot(losses, r["name"], args.output)


if __name__ == "__main__":
    main()
