#!/usr/bin/env python3
"""
GPU Test Suite Orchestrator
============================
Edit config.yaml to set node IPs, then run:

    python run_tests.py                          # all tests
    python run_tests.py --skip-per-gpu           # skip per-GPU tests
    python run_tests.py --skip-single-node       # skip single-node distributed
    python run_tests.py --skip-multi-node        # skip multi-node distributed
    python run_tests.py --nodes 0 1              # only test nodes 0 and 1

The script SSHes into each machine, deploys the tests/ directory,
runs the tests remotely, fetches results, and prints a summary.

Requirements on REMOTE machines:
  - Python 3.8+ with PyTorch (CUDA)
  - SSH access with the key configured in config.yaml
  - Passwordless sudo NOT needed
  - Port master_port open between all nodes (default 29500)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ---------------------------------------------------------------------------
# SSH / SCP helpers (async)
# ---------------------------------------------------------------------------

async def ssh(host: str, user: str, key: str, cmd: str,
              timeout: int = 600) -> Tuple[int, str, str]:
    """Run `cmd` on remote host. Returns (returncode, stdout, stderr)."""
    key = os.path.expanduser(key)
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-i", key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        f"{user}@{host}", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "", f"SSH command timed out after {timeout}s"



async def scp_from(host: str, user: str, key: str,
                   remote: str, local: str) -> Tuple[int, str]:
    """Copy remote path to local."""
    key = os.path.expanduser(key)
    proc = await asyncio.create_subprocess_exec(
        "scp", "-r", "-i", key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{user}@{host}:{remote}", local,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stderr.decode()


# ---------------------------------------------------------------------------
# Node deployment
# ---------------------------------------------------------------------------

async def deploy_to_node(node: Dict, repo_url: str, work_dir: str) -> bool:
    host = node["host"]
    user = node["user"]
    key  = node["key_file"]

    # Derive raw URL for setup_node.sh from the repo URL
    raw_base = repo_url.replace("github.com", "raw.githubusercontent.com").removesuffix(".git")
    setup_url = f"{raw_base}/main/setup_node.sh"

    print(f"  [{host}] Running setup script …")
    rc, stdout, err = await ssh(
        host, user, key,
        f"curl -fsSL {setup_url} | bash -s -- {repo_url} {work_dir}",
        timeout=300,
    )
    print(stdout, end="", flush=True)
    if rc != 0:
        print(f"  [{host}] ERROR in setup: {err}")
        return False

    print(f"  [{host}] Deployed OK")
    return True


# ---------------------------------------------------------------------------
# Per-GPU tests
# ---------------------------------------------------------------------------

async def run_per_gpu_on_node(node: Dict, gpu_ids: List[int],
                               work_dir: str, cfg: Dict,
                               results_dir: Path) -> List[Dict]:
    """Launch per_gpu.py for each GPU in parallel on one node."""
    host = node["host"]
    user = node["user"]
    key  = node["key_file"]
    cfg_json = json.dumps(cfg).replace('"', '\\"')

    async def run_one(gpu_id):
        out_remote = f"{work_dir}/per_gpu_{gpu_id}.json"
        cmd = (f"cd {work_dir} && venv/bin/python tests/per_gpu.py "
               f"--gpu-id {gpu_id} "
               f"--config \"{cfg_json}\" "
               f"--output {out_remote}")
        rc, stdout, stderr = await ssh(host, user, key, cmd, timeout=300)
        print(stdout, end="", flush=True)
        if rc != 0:
            print(f"  [{host}] GPU{gpu_id} per_gpu FAILED:\n{stderr}")
            return {"gpu_id": gpu_id, "error": stderr, "tests": [], "summary": {}}

        # Fetch result
        local_out = results_dir / f"{host}_gpu{gpu_id}.json"
        rc2, err2 = await scp_from(host, user, key, out_remote, str(local_out))
        if rc2 != 0:
            print(f"  [{host}] GPU{gpu_id} scp failed: {err2}")
            return {"gpu_id": gpu_id, "error": err2, "tests": [], "summary": {}}

        with open(local_out) as f:
            return json.load(f)

    tasks = [run_one(g) for g in gpu_ids]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Single-node distributed tests
# ---------------------------------------------------------------------------

async def run_single_node(node: Dict, gpus_per_node: int,
                           work_dir: str, cfg: Dict,
                           results_dir: Path) -> Optional[Dict]:
    host = node["host"]
    user = node["user"]
    key  = node["key_file"]
    cfg_json = json.dumps(cfg).replace('"', '\\"')
    out_remote = f"{work_dir}/single_node.json"

    cmd = (f"cd {work_dir} && "
           f"venv/bin/torchrun --standalone --nproc_per_node={gpus_per_node} "
           f"tests/single_node.py "
           f"--config \"{cfg_json}\" "
           f"--output {out_remote}")

    print(f"\n  [{host}] Running single-node tests …")
    rc, stdout, stderr = await ssh(host, user, key, cmd, timeout=600)
    print(stdout, end="", flush=True)
    if rc != 0:
        print(f"  [{host}] single_node FAILED:\n{stderr}")
        return {"host": host, "error": stderr}

    local_out = results_dir / f"{host}_single_node.json"
    rc2, err2 = await scp_from(host, user, key, out_remote, str(local_out))
    if rc2 != 0:
        return {"host": host, "error": err2}

    with open(local_out) as f:
        data = json.load(f)
    data["host"] = host
    return data


# ---------------------------------------------------------------------------
# Multi-node distributed tests
# ---------------------------------------------------------------------------

async def run_multi_node(nodes: List[Dict], gpus_per_node: int, master_port: int,
                          work_dir: str, cfg: Dict,
                          results_dir: Path) -> Optional[Dict]:
    """
    Launch torchrun on ALL nodes simultaneously.
    Node 0 is the rendezvous endpoint.
    Only node 0 writes the result file; we fetch it from there.
    """
    master_ip = nodes[0]["host"]
    n_nodes   = len(nodes)
    cfg_json  = json.dumps(cfg).replace('"', '\\"')

    async def launch_on_node(idx, node):
        host = node["host"]
        user = node["user"]
        key  = node["key_file"]
        out_remote = f"{work_dir}/multi_node.json"

        cmd = (f"cd {work_dir} && "
               f"venv/bin/torchrun "
               f"--nnodes={n_nodes} "
               f"--nproc_per_node={gpus_per_node} "
               f"--node_rank={idx} "
               f"--rdzv_backend=c10d "
               f"--rdzv_endpoint={master_ip}:{master_port} "
               f"--rdzv_id=gpu_test_suite "
               f"tests/multi_node.py "
               f"--config \"{cfg_json}\" "
               f"--output {out_remote}")

        print(f"  [{host}] Launching multi-node torchrun (node_rank={idx}) …")
        rc, stdout, stderr = await ssh(host, user, key, cmd, timeout=900)
        print(stdout, end="", flush=True)
        if rc != 0:
            print(f"  [{host}] multi_node FAILED:\n{stderr}")
        return rc, host

    print(f"\n=== Multi-Node Tests ({n_nodes} nodes × {gpus_per_node} GPUs) ===")
    tasks = [launch_on_node(i, n) for i, n in enumerate(nodes)]
    results_launch = await asyncio.gather(*tasks)

    failed_nodes = [h for rc, h in results_launch if rc != 0]
    if failed_nodes:
        print(f"  Multi-node launch failed on: {failed_nodes}")
        return {"error": f"Launch failed on {failed_nodes}"}

    # Fetch results from master node (node 0)
    master_node = nodes[0]
    out_remote = f"{work_dir}/multi_node.json"
    local_out  = results_dir / f"multi_node.json"
    rc, err = await scp_from(master_node["host"], master_node["user"],
                              master_node["key_file"], out_remote, str(local_out))
    if rc != 0:
        return {"error": f"scp from master failed: {err}"}

    with open(local_out) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

STATUS_SYM  = {"PASS": "✓", "FAIL": "✗", "WARN": "△"}
STATUS_PAD  = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}


def _sym(s):
    return STATUS_SYM.get(s, "?")


def print_section(title: str):
    print(f"\n{'━' * 70}")
    print(f"  {title}")
    print(f"{'━' * 70}")


def print_per_gpu_report(node_results: Dict[str, List[Dict]]):
    print_section("PER-GPU RESULTS")
    rows = []
    for host, gpu_list in node_results.items():
        for gpu_data in gpu_list:
            gpu_id = gpu_data.get("gpu_id", "?")
            for t in gpu_data.get("tests", []):
                rows.append([host, f"GPU{gpu_id}", _sym(t["status"]),
                              t["name"], t.get("details", t.get("error", ""))])

    if HAS_TABULATE:
        print(tabulate(rows, headers=["Node", "GPU", "St", "Test", "Details"],
                       tablefmt="simple", maxcolwidths=[20, 5, 2, 30, 60]))
    else:
        for r in rows:
            print(f"  {r[0]:<18} {r[1]} [{r[2]}] {r[3]:<35} {r[4]}")


def print_distributed_report(label: str, results_by_host: List[Dict]):
    print_section(label)
    for data in results_by_host:
        host = data.get("host", "unknown")
        if "error" in data and not data.get("tests"):
            print(f"  {host}: ERROR — {data['error']}")
            continue
        print(f"\n  Node: {host}")
        rows = []
        for t in data.get("tests", []):
            rows.append([_sym(t["status"]), t["name"], t.get("details", t.get("error", ""))])
        if HAS_TABULATE:
            print(tabulate(rows, headers=["St", "Test", "Details"],
                           tablefmt="simple", maxcolwidths=[2, 40, 65]))
        else:
            for r in rows:
                print(f"    [{r[0]}] {r[1]:<40} {r[2]}")


def print_summary(per_gpu_results, single_node_results, multi_node_result):
    print_section("SUMMARY")

    total = passed = failed = warned = 0

    def tally(tests_list):
        nonlocal total, passed, failed, warned
        for t in tests_list:
            total += 1
            s = t.get("status", "FAIL")
            if s == "PASS":   passed += 1
            elif s == "FAIL": failed += 1
            elif s == "WARN": warned += 1

    for gpu_list in per_gpu_results.values():
        for gd in gpu_list:
            tally(gd.get("tests", []))

    for data in single_node_results:
        tally(data.get("tests", []))

    if multi_node_result:
        tally(multi_node_result.get("tests", []))

    print(f"  Total tests : {total}")
    print(f"  ✓ Passed    : {passed}")
    print(f"  △ Warnings  : {warned}")
    print(f"  ✗ Failed    : {failed}")
    if failed > 0:
        print("\n  FAILED TESTS:")
        def show_fails(tests_list, prefix=""):
            for t in tests_list:
                if t.get("status") == "FAIL":
                    print(f"    {prefix} {t['name']}: {t.get('error','')[:120]}")
        for host, gpu_list in per_gpu_results.items():
            for gd in gpu_list:
                show_fails(gd.get("tests", []), prefix=f"[{host} GPU{gd.get('gpu_id','?')}]")
        for data in single_node_results:
            show_fails(data.get("tests", []), prefix=f"[{data.get('host','')} single_node]")
        if multi_node_result:
            show_fails(multi_node_result.get("tests", []), prefix="[multi_node]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args, cfg):
    nodes          = cfg["nodes"]
    gpus_per_node  = cfg.get("gpus_per_node", 8)
    master_port    = cfg.get("master_port", 29500)
    work_dir       = cfg.get("remote_work_dir", "~/gpu_full_test")
    repo_url       = cfg["repo_url"]
    test_cfg       = cfg.get("tests", {})
    results_root   = Path(cfg.get("results_dir", "./results"))

    # Filter nodes if requested
    if args.nodes:
        nodes = [nodes[i] for i in args.nodes if i < len(nodes)]

    # Timestamped results dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = results_root / ts
    results_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = list(range(gpus_per_node))

    # ── Step 1: Deploy ──────────────────────────────────────────────────────
    print_section(f"DEPLOYING TO {len(nodes)} NODE(S)")
    deploy_tasks = [deploy_to_node(n, repo_url, work_dir) for n in nodes]
    deploy_ok = await asyncio.gather(*deploy_tasks)
    if not all(deploy_ok):
        failed = [n["host"] for n, ok in zip(nodes, deploy_ok) if not ok]
        print(f"\nERROR: Deploy failed on {failed}. Aborting.")
        sys.exit(1)

    per_gpu_results    = {}   # host -> List[gpu_result]
    single_node_results = []
    multi_node_result   = None

    # ── Step 2: Per-GPU tests ───────────────────────────────────────────────
    if not args.skip_per_gpu:
        print_section(f"PER-GPU TESTS ({len(nodes)} nodes × {gpus_per_node} GPUs)")
        node_tasks = [
            run_per_gpu_on_node(n, gpu_ids, work_dir, test_cfg, results_dir)
            for n in nodes
        ]
        node_results_list = await asyncio.gather(*node_tasks)
        for node, gpu_results in zip(nodes, node_results_list):
            per_gpu_results[node["host"]] = gpu_results
    else:
        print("  [skipped]")

    # ── Step 3: Single-node tests ───────────────────────────────────────────
    if not args.skip_single_node:
        print_section(f"SINGLE-NODE TESTS ({len(nodes)} nodes in parallel)")
        sn_tasks = [
            run_single_node(n, gpus_per_node, work_dir, test_cfg, results_dir)
            for n in nodes
        ]
        sn_results = await asyncio.gather(*sn_tasks)
        single_node_results = [r for r in sn_results if r is not None]
    else:
        print("  [skipped]")

    # ── Step 4: Multi-node tests ────────────────────────────────────────────
    if not args.skip_multi_node and len(nodes) > 1:
        multi_node_result = await run_multi_node(
            nodes, gpus_per_node, master_port, work_dir, test_cfg, results_dir
        )
    elif args.skip_multi_node:
        print("\n  [multi-node skipped]")
    else:
        print("\n  [multi-node skipped — need ≥2 nodes]")

    # ── Step 5: Report ──────────────────────────────────────────────────────
    if per_gpu_results:
        print_per_gpu_report(per_gpu_results)

    if single_node_results:
        print_distributed_report("SINGLE-NODE DISTRIBUTED RESULTS", single_node_results)

    if multi_node_result and "tests" in multi_node_result:
        print_distributed_report("MULTI-NODE DISTRIBUTED RESULTS", [multi_node_result])

    print_summary(per_gpu_results, single_node_results, multi_node_result or {})
    print(f"\n  Full JSON results saved to: {results_dir.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="GPU Test Suite Orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-per-gpu",     action="store_true")
    parser.add_argument("--skip-single-node", action="store_true")
    parser.add_argument("--skip-multi-node",  action="store_true")
    parser.add_argument("--nodes", nargs="*", type=int,
                        help="Indices of nodes to test (default: all). E.g. --nodes 0 1")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("nodes"):
        print("ERROR: no nodes defined in config.yaml")
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"  GPU Test Suite — {len(cfg['nodes'])} nodes × {cfg.get('gpus_per_node', 8)} GPUs")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}")

    asyncio.run(main_async(args, cfg))


if __name__ == "__main__":
    main()
