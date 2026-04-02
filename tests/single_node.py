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
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

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


def all_gather_scalar(val: float, rank: int, world: int) -> List[float]:
    t = torch.tensor([val], dtype=torch.float64, device=f"cuda:{rank}")
    gathered = [torch.zeros(1, dtype=torch.float64, device=f"cuda:{rank}") for _ in range(world)]
    dist.all_gather(gathered, t)
    return [g.item() for g in gathered]


# ---------------------------------------------------------------------------
# Test: NVLink P2P bandwidth (ring)
# ---------------------------------------------------------------------------

def test_nvlink_ring_bandwidth(rank: int, world: int, data_gb: float = 4.0) -> Optional[Dict]:
    """
    Simultaneous ring send/recv: rank r → rank (r+1)%world at the same time.
    Measures unidirectional NVLink throughput per link.
    H200 NVLink 4.0: ~450 GB/s uni per GPU. Threshold: 200 GB/s (conservative,
    allows for NCCL overhead and shared NVSwitch fabric).
    """
    try:
        device = f"cuda:{rank}"
        n = int(data_gb * 1024**3) // 4  # float32 elements
        send_to   = (rank + 1) % world
        recv_from = (rank - 1) % world

        send_t = torch.ones(n, dtype=torch.float32, device=device)
        recv_t = torch.empty(n, dtype=torch.float32, device=device)

        # Warmup
        s = dist.isend(send_t[:4096], dst=send_to)
        r = dist.irecv(recv_t[:4096], src=recv_from)
        s.wait(); r.wait()
        barrier_sync(rank, world)

        iters = 5
        barrier_sync(rank, world)
        t0 = time.perf_counter()
        for _ in range(iters):
            s = dist.isend(send_t, dst=send_to)
            r = dist.irecv(recv_t, src=recv_from)
            s.wait(); r.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        barrier_sync(rank, world)

        bw_per_rank = data_gb * iters / elapsed  # GB/s unidirectional

        all_bws = all_gather_scalar(bw_per_rank, rank, world)
        del send_t, recv_t
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        avg_bw = sum(all_bws) / len(all_bws)
        min_bw = min(all_bws)
        threshold = 200.0  # GB/s
        status = PASS if min_bw >= threshold else WARN
        return result("nvlink_ring_bandwidth", status,
                      metrics={"avg_gbps": round(avg_bw, 1),
                                "min_gbps": round(min_bw, 1),
                                "per_rank_gbps": [round(b, 1) for b in all_bws],
                                "threshold_gbps": threshold,
                                "data_gb": data_gb},
                      details=f"avg {avg_bw:.1f} GB/s | min {min_bw:.1f} GB/s | threshold {threshold} GB/s")
    except Exception:
        if rank == 0:
            return result("nvlink_ring_bandwidth", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: NVLink all-pairs bandwidth matrix
# ---------------------------------------------------------------------------

def test_nvlink_all_pairs(rank: int, world: int, data_gb: float = 1.0) -> Optional[Dict]:
    """
    Tests every (src, dst) pair one at a time. Detects broken/asymmetric links.
    In an NVSwitch topology all pairs should be within ~10% of each other.
    """
    try:
        device = f"cuda:{rank}"
        n = int(data_gb * 1024**3) // 4
        send_t = torch.ones(n, dtype=torch.float32, device=device)
        recv_t = torch.empty(n, dtype=torch.float32, device=device)

        bw_matrix = {}  # (src, dst) -> GB/s

        for src in range(world):
            for dst in range(world):
                if src == dst:
                    continue
                barrier_sync(rank, world)
                iters = 2
                t0 = time.perf_counter()
                for _ in range(iters):
                    if rank == src:
                        dist.send(send_t, dst=dst)
                    elif rank == dst:
                        dist.recv(recv_t, src=src)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                barrier_sync(rank, world)

                if rank == src:
                    bw = data_gb * iters / elapsed
                    bw_tensor = torch.tensor([bw], dtype=torch.float64, device=device)
                else:
                    bw_tensor = torch.zeros(1, dtype=torch.float64, device=device)

                # Broadcast result from src to rank 0
                dist.broadcast(bw_tensor, src=src)
                bw_matrix[(src, dst)] = bw_tensor.item()

        del send_t, recv_t
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        bws = list(bw_matrix.values())
        avg = sum(bws) / len(bws)
        minimum = min(bws)
        maximum = max(bws)
        # Flag any link more than 30% below average
        bad_links = [(f"{s}→{d}", round(b, 1))
                     for (s, d), b in bw_matrix.items() if b < avg * 0.7]

        status = PASS if not bad_links else WARN
        return result("nvlink_all_pairs", status,
                      metrics={"avg_gbps": round(avg, 1),
                                "min_gbps": round(minimum, 1),
                                "max_gbps": round(maximum, 1),
                                "bad_links": bad_links,
                                "bandwidth_matrix": {f"{s}→{d}": round(b, 1)
                                                     for (s, d), b in bw_matrix.items()}},
                      details=f"avg {avg:.1f} | min {minimum:.1f} | max {maximum:.1f} GB/s"
                              + (f" | SLOW LINKS: {bad_links}" if bad_links else " | all links symmetric"))
    except Exception:
        if rank == 0:
            return result("nvlink_all_pairs", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: NVLink P2P latency
# ---------------------------------------------------------------------------

def test_nvlink_latency(rank: int, world: int) -> Optional[Dict]:
    """
    Ping-pong latency between GPU 0 and GPU 1 (smallest message: 1 float).
    Measures round-trip and reports one-way latency.
    """
    try:
        device = f"cuda:{rank}"
        tiny = torch.zeros(1, dtype=torch.float32, device=device)

        if rank not in (0, 1):
            barrier_sync(rank, world)
            return None

        # Warmup
        for _ in range(20):
            if rank == 0:
                dist.send(tiny, dst=1)
                dist.recv(tiny, src=1)
            else:
                dist.recv(tiny, src=0)
                dist.send(tiny, dst=0)

        barrier_sync(rank, world)

        iters = 200
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            if rank == 0:
                dist.send(tiny, dst=1)
                dist.recv(tiny, src=1)
            else:
                dist.recv(tiny, src=0)
                dist.send(tiny, dst=0)
        torch.cuda.synchronize()
        rtt_us = (time.perf_counter() - t0) / iters * 1e6  # µs round-trip
        one_way_us = rtt_us / 2

        barrier_sync(rank, world)

        if rank != 0:
            return None

        # H200 NVLink latency: ~1-3 µs one-way typical
        threshold_us = 10.0
        status = PASS if one_way_us < threshold_us else WARN
        return result("nvlink_latency", status,
                      metrics={"one_way_us": round(one_way_us, 2),
                                "round_trip_us": round(rtt_us, 2),
                                "threshold_us": threshold_us},
                      details=f"one-way {one_way_us:.2f} µs | RTT {rtt_us:.2f} µs | threshold {threshold_us} µs")
    except Exception:
        if rank == 0:
            return result("nvlink_latency", FAIL, error=traceback.format_exc())
        return None


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
# Test: AllReduce bandwidth sweep
# ---------------------------------------------------------------------------

def test_allreduce_bandwidth(rank: int, world: int, sizes_mb: List[int] = None) -> Optional[Dict]:
    """
    Ring-AllReduce bus bandwidth at multiple payload sizes.
    Bus BW formula: (2*(N-1)/N) * size / time
    H200 NVLink AllReduce: ~200-400 GB/s bus BW for large messages.
    Threshold: 100 GB/s (conservative, accounts for NCCL tuning).
    """
    if sizes_mb is None:
        sizes_mb = [64, 256, 1024, 4096]

    try:
        device = f"cuda:{rank}"
        bw_results = {}

        for size_mb in sizes_mb:
            n = int(size_mb * 1024**2) // 4  # float32
            t = torch.ones(n, dtype=torch.float32, device=device)

            # Warmup
            for _ in range(3):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            barrier_sync(rank, world)

            iters = 3
            torch.cuda.synchronize()
            t0_wall = time.perf_counter()
            for _ in range(iters):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            barrier_sync(rank, world)
            elapsed = time.perf_counter() - t0_wall

            size_bytes = size_mb * 1024**2
            # Standard bus bandwidth formula for ring allreduce
            bus_bw_gbps = 2 * (world - 1) / world * size_bytes * iters / elapsed / 1e9
            bw_results[size_mb] = round(bus_bw_gbps, 1)
            del t
            torch.cuda.empty_cache()

        if rank != 0:
            return None

        min_bw = min(bw_results.values())
        threshold = 100.0  # GB/s bus BW
        status = PASS if min_bw >= threshold else WARN
        details = " | ".join(f"{s}MB→{b}GB/s" for s, b in bw_results.items())
        return result("allreduce_bandwidth", status,
                      metrics={"by_size_mb_gbps": bw_results,
                                "min_gbps": min_bw,
                                "threshold_gbps": threshold},
                      details=details)
    except Exception:
        if rank == 0:
            return result("allreduce_bandwidth", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: AllGather bandwidth
# ---------------------------------------------------------------------------

def test_allgather_bandwidth(rank: int, world: int, size_mb: int = 512) -> Optional[Dict]:
    try:
        device = f"cuda:{rank}"
        n = int(size_mb * 1024**2) // 4
        send_t = torch.ones(n, dtype=torch.float32, device=device)
        recv_ts = [torch.empty(n, dtype=torch.float32, device=device) for _ in range(world)]

        for _ in range(3):
            dist.all_gather(recv_ts, send_t)
        barrier_sync(rank, world)

        iters = 3
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_gather(recv_ts, send_t)
        torch.cuda.synchronize()
        barrier_sync(rank, world)
        elapsed = time.perf_counter() - t0

        # Each rank receives (world-1)*size_mb, sends size_mb
        total_data_gb = world * size_mb * 1024**2 * iters / 1e9
        bw_gbps = total_data_gb / elapsed / world  # per-GPU algbw

        del send_t, recv_ts
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        threshold = 50.0
        status = PASS if bw_gbps >= threshold else WARN
        return result("allgather_bandwidth", status,
                      metrics={"algbw_gbps": round(bw_gbps, 1), "size_mb": size_mb,
                                "threshold_gbps": threshold},
                      details=f"algbw {bw_gbps:.1f} GB/s for {size_mb}MB payload")
    except Exception:
        if rank == 0:
            return result("allgather_bandwidth", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: ReduceScatter bandwidth
# ---------------------------------------------------------------------------

def test_reduce_scatter_bandwidth(rank: int, world: int, size_mb: int = 512) -> Optional[Dict]:
    try:
        device = f"cuda:{rank}"
        n_per_rank = int(size_mb * 1024**2) // 4
        n_total = n_per_rank * world
        send_t = torch.ones(n_total, dtype=torch.float32, device=device)
        recv_t = torch.empty(n_per_rank, dtype=torch.float32, device=device)

        for _ in range(3):
            dist.reduce_scatter_tensor(recv_t, send_t)
        barrier_sync(rank, world)

        iters = 3
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.reduce_scatter_tensor(recv_t, send_t)
        torch.cuda.synchronize()
        barrier_sync(rank, world)
        elapsed = time.perf_counter() - t0

        bw_gbps = n_total * 4 * iters / elapsed / 1e9  # total bytes / time / world
        bw_gbps /= world

        del send_t, recv_t
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        threshold = 50.0
        status = PASS if bw_gbps >= threshold else WARN
        return result("reduce_scatter_bandwidth", status,
                      metrics={"algbw_gbps": round(bw_gbps, 1), "size_mb": size_mb,
                                "threshold_gbps": threshold},
                      details=f"algbw {bw_gbps:.1f} GB/s for {size_mb}MB payload")
    except Exception:
        if rank == 0:
            return result("reduce_scatter_bandwidth", FAIL, error=traceback.format_exc())
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
    2. Verifies loss decreases (model is actually learning).
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

        # Linear relationship y = X @ w guarantees loss decreases
        torch.manual_seed(0)
        w = torch.randn(hidden, 1, device=device)
        X = torch.randn(batch_size, hidden, device=device)
        y = X @ w

        first_loss = None
        last_loss = None
        grad_sync_ok = True

        barrier_sync(rank, world)
        t0 = time.perf_counter()

        for step in range(steps):
            optimizer.zero_grad()
            out = ddp_model(X)
            loss = criterion(out, y)
            loss.backward()

            # Verify gradient synchronisation every 10 steps
            if step % 10 == 0:
                # All ranks must have the same gradient norm for the first linear layer
                gn = ddp_model.module.net[0].weight.grad.norm().item()
                gn_tensor = torch.tensor([gn], device=device)
                dist.all_reduce(gn_tensor, op=dist.ReduceOp.MAX)
                max_gn = gn_tensor.item()
                dist.all_reduce(torch.tensor([gn], device=device), op=dist.ReduceOp.MIN)
                if max_gn > 0 and abs(gn - max_gn) / max_gn > 0.001:
                    grad_sync_ok = False

            optimizer.step()

            lv = loss.item()
            if first_loss is None:
                first_loss = lv
            last_loss = lv

        torch.cuda.synchronize()
        barrier_sync(rank, world)
        elapsed = time.perf_counter() - t0

        throughput = batch_size * steps * world / elapsed

        del ddp_model, model, optimizer, X, y, w
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        loss_decreased = last_loss < first_loss
        status = PASS if (loss_decreased and grad_sync_ok) else FAIL
        return result("ddp_training_single_node", status,
                      metrics={"first_loss": round(first_loss, 6),
                                "last_loss": round(last_loss, 6),
                                "loss_decreased": loss_decreased,
                                "grad_sync_ok": grad_sync_ok,
                                "throughput_samples_per_s": round(throughput, 1),
                                "steps": steps,
                                "world_size": world},
                      details=f"loss {first_loss:.4f}→{last_loss:.4f} | "
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

    if rank == 0:
        print(f"\n=== Single-Node Tests (world_size={world}) ===")

    run(lambda: test_nvlink_ring_bandwidth(rank, world, cfg.get("nvlink_data_gb", 1.0)))
    run(lambda: test_nvlink_latency(rank, world))
    run(lambda: test_allreduce_correctness(rank, world))
    run(lambda: test_allreduce_bandwidth(rank, world, cfg.get("allreduce_sizes_mb", [64, 256, 1024, 4096])))
    run(lambda: test_allgather_bandwidth(rank, world))
    run(lambda: test_reduce_scatter_bandwidth(rank, world))
    run(lambda: test_ddp_training(rank, world,
                                   hidden=cfg.get("ddp_hidden_size", 4096),
                                   steps=cfg.get("ddp_steps", 50)))

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


if __name__ == "__main__":
    main()
