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
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

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
    rank  = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank  = int(os.environ.get("NODE_RANK", 0))
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

def test_allreduce_correctness(rank, world, local_rank, node_rank):
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

def test_allreduce_bandwidth(rank, world, local_rank, sizes_mb=None):
    """
    Measures inter-node AllReduce bus bandwidth.
    For 3 nodes × 8 GPUs = 24 ranks. Threshold: 20 GB/s bus BW
    (network-limited at 400 Gb/s IB HDR; 20 GB/s is ~40% utilisation of 50 GB/s line rate).
    """
    if sizes_mb is None:
        sizes_mb = [64, 256, 1024, 4096]

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
        threshold = 20.0  # GB/s bus BW (inter-node, network-limited)
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
# Test: Cross-node AllGather bandwidth
# ---------------------------------------------------------------------------

def test_allgather_bandwidth(rank, world, local_rank, size_mb=512):
    try:
        dev = f"cuda:{local_rank}"
        n = int(size_mb * 1024**2) // 4
        send_t = torch.ones(n, dtype=torch.float32, device=dev)
        recv_ts = [torch.empty(n, dtype=torch.float32, device=dev) for _ in range(world)]

        for _ in range(3):
            dist.all_gather(recv_ts, send_t)
        barrier_sync(rank, world)

        iters = 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_gather(recv_ts, send_t)
        torch.cuda.synchronize()
        barrier_sync(rank, world)
        elapsed = time.perf_counter() - t0

        algbw = size_mb * 1024**2 * (world - 1) / world * iters / elapsed / 1e9

        del send_t, recv_ts
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        threshold = 10.0
        status = PASS if algbw >= threshold else WARN
        return result("allgather_bandwidth_multi_node", status,
                      metrics={"algbw_gbps": round(algbw, 1), "size_mb": size_mb,
                                "threshold_gbps": threshold},
                      details=f"algbw {algbw:.1f} GB/s for {size_mb}MB payload")
    except Exception:
        if rank == 0:
            return result("allgather_bandwidth_multi_node", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: Cross-node ReduceScatter bandwidth
# ---------------------------------------------------------------------------

def test_reduce_scatter_bandwidth(rank, world, local_rank, size_mb=512):
    try:
        dev = f"cuda:{local_rank}"
        n_per_rank = int(size_mb * 1024**2) // 4
        n_total = n_per_rank * world
        send_t = torch.ones(n_total, dtype=torch.float32, device=dev)
        recv_t = torch.empty(n_per_rank, dtype=torch.float32, device=dev)

        for _ in range(3):
            dist.reduce_scatter_tensor(recv_t, send_t)
        barrier_sync(rank, world)

        iters = 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.reduce_scatter_tensor(recv_t, send_t)
        torch.cuda.synchronize()
        barrier_sync(rank, world)
        elapsed = time.perf_counter() - t0

        algbw = n_per_rank * 4 * iters / elapsed / 1e9

        del send_t, recv_t
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        threshold = 10.0
        status = PASS if algbw >= threshold else WARN
        return result("reduce_scatter_bandwidth_multi_node", status,
                      metrics={"algbw_gbps": round(algbw, 1), "size_mb": size_mb,
                                "threshold_gbps": threshold},
                      details=f"algbw {algbw:.1f} GB/s for {size_mb}MB payload")
    except Exception:
        if rank == 0:
            return result("reduce_scatter_bandwidth_multi_node", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: Cross-node point-to-point bandwidth (node 0 → node 1)
# ---------------------------------------------------------------------------

def test_cross_node_p2p_bandwidth(rank, world, local_rank, node_rank, gpus_per_node,
                                   data_gb=4.0):
    """
    Rank 0 (node 0, GPU 0) sends large tensor to rank gpus_per_node (node 1, GPU 0).
    Measures unidirectional cross-node network bandwidth.
    """
    try:
        dev = f"cuda:{local_rank}"
        n = int(data_gb * 1024**3) // 4
        src_rank = 0
        dst_rank = gpus_per_node  # First GPU on node 1

        if rank not in (src_rank, dst_rank):
            barrier_sync(rank, world)
            return None

        send_t = torch.ones(n, dtype=torch.float32, device=dev) if rank == src_rank else None
        recv_t = torch.empty(n, dtype=torch.float32, device=dev) if rank == dst_rank else None

        # Warmup
        iters_warm = 2
        for _ in range(iters_warm):
            if rank == src_rank:
                dist.send(send_t[:1024], dst=dst_rank)
            else:
                dist.recv(recv_t[:1024], src=src_rank)

        barrier_sync(rank, world)

        iters = 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            if rank == src_rank:
                dist.send(send_t, dst=dst_rank)
            else:
                dist.recv(recv_t, src=src_rank)
        torch.cuda.synchronize()
        barrier_sync(rank, world)
        elapsed = time.perf_counter() - t0

        del send_t, recv_t
        torch.cuda.empty_cache()

        if rank != 0:
            return None

        bw_gbps = data_gb * iters / elapsed
        # 400 Gb/s InfiniBand = 50 GB/s; 200 Gb/s HDR = 25 GB/s
        threshold = 20.0
        status = PASS if bw_gbps >= threshold else WARN
        return result("cross_node_p2p_bandwidth", status,
                      metrics={"gbps": round(bw_gbps, 2), "threshold_gbps": threshold,
                                "data_gb": data_gb},
                      details=f"node0→node1: {bw_gbps:.2f} GB/s | threshold {threshold} GB/s")
    except Exception:
        if rank == 0:
            return result("cross_node_p2p_bandwidth", FAIL, error=traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Test: Barrier latency (measures synchronisation overhead)
# ---------------------------------------------------------------------------

def test_barrier_latency(rank, world):
    try:
        # Warmup
        for _ in range(20):
            dist.barrier()
        torch.cuda.synchronize()

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.barrier()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        latency_ms = elapsed / iters * 1000

        if rank != 0:
            return None

        threshold_ms = 10.0  # 10 ms is generous for 3-node cluster
        status = PASS if latency_ms < threshold_ms else WARN
        return result("barrier_latency", status,
                      metrics={"latency_ms": round(latency_ms, 3),
                                "threshold_ms": threshold_ms,
                                "iters": iters},
                      details=f"barrier latency {latency_ms:.3f} ms | threshold {threshold_ms} ms")
    except Exception:
        if rank == 0:
            return result("barrier_latency", FAIL, error=traceback.format_exc())
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


def test_ddp_training(rank, world, local_rank, hidden=4096, steps=50, batch_size=64):
    """
    Multi-node DDP training loop. Same checks as single-node:
    - Loss must decrease
    - Gradient norms must be identical across ALL ranks (cross-node allreduce)
    - Reports cross-node throughput
    """
    try:
        device = torch.device(f"cuda:{local_rank}")
        torch.manual_seed(42)

        model = SimpleMLP(hidden).to(device)
        ddp_model = DDP(model, device_ids=[local_rank])
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
        return result("ddp_training_multi_node", status,
                      metrics={"first_loss": round(first_loss, 6),
                                "last_loss": round(last_loss, 6),
                                "loss_decreased": loss_decreased,
                                "grad_sync_ok": grad_sync_ok,
                                "throughput_samples_per_s": round(throughput, 1),
                                "world_size": world,
                                "nodes": world // 8},
                      details=f"loss {first_loss:.4f}→{last_loss:.4f} | "
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

    run(lambda: test_allreduce_correctness(rank, world, local_rank, node_rank))
    run(lambda: test_ddp_training(rank, world, local_rank,
                                   hidden=cfg.get("ddp_hidden_size", 4096),
                                   steps=cfg.get("ddp_steps", 20)))

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


if __name__ == "__main__":
    main()
