# GPU Test Suite

Automated acceptance tests for multi-GPU / multi-node clusters. Verifies compute correctness, memory health, bandwidth, and distributed training across all GPUs.

## What it tests

| Scope | Tests |
|-------|-------|
| **Per-GPU** | GPU info, MatMul correctness, memory allocation, PCIe bandwidth, HBM bandwidth, BF16 TFLOPS, mixed-precision correctness, single-GPU training |
| **Single-node** | AllReduce correctness, DDP training (gradient sync + loss convergence) |
| **Multi-node** | Cross-node AllReduce correctness, AllReduce bandwidth, DDP training |

Results are saved as JSON + loss curve PNG to `./results/<timestamp>/`.

---

## Setup

### 1. Configure your cluster

Edit `config.yaml`:

```yaml
nodes:
  - host: "10.0.0.1"        # SSH-reachable IP of each node
    user: "ubuntu"
    key_file: "~/.ssh/id_ed25519"
  - host: "10.0.0.2"
    user: "ubuntu"
    key_file: "~/.ssh/id_ed25519"

gpus_per_node: 8
master_port: 29500
remote_work_dir: "~/gpu_full_test"
repo_url: "https://github.com/BMehron/gpu_full_test.git"
```

### 2. Install local dependencies

```bash
pip install pyyaml tabulate
```

### 3. Make sure SSH key access works

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<node-ip> "echo ok"
```

The orchestrator SSHes into each node automatically — no manual setup needed on the nodes.

---

## Running from your local machine

```bash
# All tests (deploy → per-GPU → single-node → multi-node)
python run_tests.py

# Skip specific stages
python run_tests.py --skip-per-gpu
python run_tests.py --skip-single-node
python run_tests.py --skip-multi-node

# Test only specific nodes (by index)
python run_tests.py --nodes 0 1
```

The orchestrator will:
1. SSH into each node and clone/update the repo
2. Create a Python venv and install dependencies
3. Run the tests remotely and stream output line-by-line as it arrives
4. Fetch all result files (JSON + loss plots) to `./results/<timestamp>/`
5. Clean up the remote repo after tests complete

---

## Tuning thresholds

All pass/fail thresholds are in `config.yaml` under `tests:`. Adjust them to match your hardware:

```yaml
tests:
  # Hardware specs
  h200_peak_bf16_tflops: 989.0      # Dense BF16 peak for your GPU model

  # Pass/warn thresholds
  tflops_threshold_pct: 65.0        # % of rated peak required to PASS
  hbm_threshold_tbps: 2.5           # HBM bandwidth floor (TB/s)
  pcie_threshold_gbps: 25.0         # PCIe bandwidth floor (GB/s)
  allreduce_threshold_gbps: 2.0     # Inter-node allreduce floor (GB/s); raise for IB clusters

  # Training test
  ddp_batch_size: 64                # Batch size per GPU
  ddp_steps: 100                    # Training steps
  ddp_loss_drop_threshold: 0.5      # Minimum relative loss drop (0.5 = 50%)
```

---

## Results

After each run, `./results/<timestamp>/` contains:

```
results/20260402_120000/
  216.48.190.157_gpu0.json        # Per-GPU results
  216.48.190.157_gpu0_loss.png    # Training loss curve
  ...
  216.48.190.157_single_node.json
  216.48.190.157_single_node_loss.png
  multi_node.json
  multi_node_loss.png
```

The terminal also prints a live summary as tests complete and a final pass/warn/fail table.
