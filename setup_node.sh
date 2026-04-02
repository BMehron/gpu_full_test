#!/usr/bin/env bash
# Usage: bash setup_node.sh <repo_url> <work_dir>
# Clones/updates the repo, creates a venv, and installs dependencies.
set -euo pipefail

REPO_URL="${1:?Usage: setup_node.sh <repo_url> <work_dir>}"
WORK_DIR="${2:?Usage: setup_node.sh <repo_url> <work_dir>}"

echo "[setup] Cloning/updating repo..."
if [ -d "$WORK_DIR/.git" ]; then
    git -C "$WORK_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$WORK_DIR"
fi

echo "[setup] Creating venv..."
[ -d "$WORK_DIR/venv" ] || python3 -m venv "$WORK_DIR/venv"

echo "[setup] Activating venv..."
source "$WORK_DIR/venv/bin/activate

echo "[setup] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$WORK_DIR/requirements.txt"

echo "[setup] Done."
