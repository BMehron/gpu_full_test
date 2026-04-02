#!/usr/bin/env python3
"""
Single-node distributed tests across all GPUs on one machine.

Launched by the orchestrator via:
    torchrun --standalone --nproc_per_node=8 tests/single_node.py \
             --config '...' --output /tmp/node_single.json

All ranks participate; rank 0 writes the JSON results file.
"""

import argparse
import json
import os
import sys
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

def result(name: str, status: str, metrics: Dict = None, details: str = "", error: str = "") -> Dict:
    return {"name": name, "status": status, "metrics": metrics or {}, "details": details, "error": error}


def init_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world


def barrier_sync(rank, world):
    dist.barrier()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Test: AllReduce correctness
# ---------------------------------------------------------------------------

def test_allreduce_correctness(rank: int, world: int) -> Optional[Dict]:
    """
    Each rank contributes rank+1 to a sum. Expected result = world*(world+1)/2.
    Checks against reference for both float32 and bfloat16.
    """
    try:
        device = f"cuda:{rank}"
        errors = []

        for dtype, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
            val = float(rank + 1)
            expected = world * (world + 1) / 2
            t = torch.full((1024,), val, dtype=dtype, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            got = t[0].float().item()
            rel_err = abs(got - expected) / expected
            tol = 0.001 if dtype == torch.float32 else 0.01
            if rel_err > tol:
                errors.append(f"{name}: expected {expected:.1f} got {got:.4f} rel_err={rel_err:.2e}")

        if rank != 0:
            return None

        status = PASS if not errors else FAIL
        return result("allreduce_correctness", status,
                      metrics={"errors": errors, "world_size": world},
                      details="fp32 + bf16 allreduce correct" if not errors else f"ERRORS: {errors}")
    except Exception:
        if rank == 0:
            return result("allreduce_correctness", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: DDP training loop (single node)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, hidden: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_ddp_training(rank: int, world: int, hidden: int = 4096, steps: int = 50,
                      batch_size: int = 64) -> Optional[Dict]:
    """
    Full DDP training loop:
    1. Runs `steps` forward+backward+optimizer steps.
    2. Verifies loss drops by ≥50% (model is actually learning).
    3. Checks gradient synchronisation: all ranks must have identical grad norms.
    4. Reports throughput (samples/s).
    """
    try:
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42 + rank)

        model = SimpleMLP(hidden).to(device)
        ddp_model = DDP(model, device_ids=[rank])
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        torch.manual_seed(0)                                      # same target function on all ranks
        w = torch.randn(hidden, 1, device=device)
        torch.manual_seed(100 + rank)                             # different samples per rank
        X = torch.randn(batch_size, hidden, device=device)
        y = X @ w

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

            # Verify gradient synchronisation every 10 steps
            if step % 10 == 0:
                gn = ddp_model.module.net[0].weight.grad.norm().item()
                max_gn_t = torch.tensor([gn], device=device)
                min_gn_t = torch.tensor([gn], device=device)
                dist.all_reduce(max_gn_t, op=dist.ReduceOp.MAX)
                dist.all_reduce(min_gn_t, op=dist.ReduceOp.MIN)
                max_gn = max_gn_t.item()
                min_gn = min_gn_t.item()
                if max_gn > 0 and (max_gn - min_gn) / max_gn > 0.001:
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
        status = PASS if (relative_drop >= 0.5 and grad_sync_ok) else FAIL
        return result("ddp_training_single_node", status,
                      metrics={"first_loss": round(first_loss, 6),
                                "last_loss": round(last_loss, 6),
                                "relative_drop_pct": round(relative_drop * 100, 1),
                                "grad_sync_ok": grad_sync_ok,
                                "throughput_samples_per_s": round(throughput, 1),
                                "steps": steps,
                                "world_size": world,
                                "step_losses": losses},
                      details=f"loss {first_loss:.4f}→{last_loss:.4f} ({relative_drop*100:.1f}% drop) | "
                              f"grad_sync {'OK' if grad_sync_ok else 'FAIL'} | "
                              f"{throughput:.0f} samples/s")
    except Exception:
        if rank == 0:
            return result("ddp_training_single_node", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="{}")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rank, world = init_dist()
    cfg = json.loads(args.config)

    if rank == 0:
        print(f"\n=== Single-Node Tests (world_size={world}) ===", flush=True)
        print("  Initializing NCCL …", flush=True)
    _w = torch.zeros(1, device=f"cuda:{rank}")
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
            print(f"  [{tag}] {r['name']:<40} {msg}", flush=True)
        return r

    run(lambda: test_allreduce_correctness(rank, world))
    run(lambda: test_ddp_training(rank, world,
                                   hidden=cfg.get("ddp_hidden_size", 4096),
                                   steps=cfg.get("ddp_steps", 20)))

    dist.destroy_process_group()

    if rank == 0:
        passed  = sum(1 for r in all_results if r["status"] == PASS)
        failed  = sum(1 for r in all_results if r["status"] == FAIL)
        warned  = sum(1 for r in all_results if r["status"] == WARN)
        output = {
            "type": "single_node",
            "world_size": world,
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
