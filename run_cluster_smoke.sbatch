#!/bin/bash
#SBATCH --job-name=marl-smoke
#SBATCH --partition=common
#SBATCH --qos=jwozniak_common
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=cluster_logs/smoke_%j.out
#SBATCH --error=cluster_logs/smoke_%j.err

# Smoke test: tiny grid to verify everything runs end-to-end on the cluster.
# Should finish in ~5 minutes.

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      $(hostname)"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Working:   $(pwd)"
nvidia-smi || true
echo "================"

mkdir -p results logs cluster_logs

if command -v uv >/dev/null 2>&1; then
    PY="uv run python"
else
    PY="python"
fi

# Point dynamic linker at cudnn/nccl shipped in the venv.
VENV_NV="$(uv run python -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
export LD_LIBRARY_PATH="${VENV_NV}/cudnn/lib:${VENV_NV}/nccl/lib:${LD_LIBRARY_PATH:-}"

$PY -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

$PY run_grid.py \
    --envs hallway \
    --mixers vdn \
    --explorations independent correlated \
    --similarities obs \
    --seeds 0 1 \
    --episodes 200 \
    --concurrency 2 \
    --out-dir results_smoke --log-dir logs_smoke \
    --parameter-sharing

echo "=== Smoke OK. Files produced: ==="
ls results_smoke/
