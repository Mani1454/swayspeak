#!/usr/bin/env bash
# One-click startup for macOS/Linux: creates/updates conda env, installs deps, runs the app.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f environment.yml ]]; then
  echo "[ERROR] environment.yml not found in $ROOT_DIR" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Install Anaconda/Miniconda and retry." >&2
  exit 1
fi

echo "[INFO] Creating/updating conda environment auraspeak..."
conda env create --file environment.yml --name auraspeak --force

# Activate conda in a POSIX shell
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate auraspeak

echo "[INFO] Installing/updating Python dependencies..."
pip install -r requirements.txt

echo "[INFO] Starting server at http://127.0.0.1:8000 ..."
cd code
python server.py
