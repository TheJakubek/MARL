#!/bin/bash
#SBATCH --job-name=marl-coop-exp
#SBATCH --partition=common
#SBATCH --qos=jwozniak_common
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cluster_logs/slurm_%j.out
#SBATCH --error=cluster_logs/slurm_%j.err

# ----------------------------------------------------------------------------
# MARL coop exploration: full grid on Entropy.
# Two sub-grids (Switches dropped — already validated locally):
#   - hallway:  fast, 1500 ep, 5 seeds
#   - lbf:      hard, 3000 ep, 5 seeds
#
# Run from project root:
#   mkdir -p cluster_logs results logs
#   sbatch run_cluster.sbatch
# ----------------------------------------------------------------------------

set -euo pipefail

# Show environment for debugging.
echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      $(hostname)"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Date:      $(date)"
echo "Working:   $(pwd)"
nvidia-smi || true
echo "================"

# Make sure output dirs exist.
mkdir -p results logs cluster_logs

# Use uv if available, fall back to python directly.
if command -v uv >/dev/null 2>&1; then
    PY="uv run python"
else
    PY="python"
fi

# Point dynamic linker at cudnn/nccl shipped in the venv so torch can find them.
VENV_NV="$(uv run python -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
export LD_LIBRARY_PATH="${VENV_NV}/cudnn/lib:${VENV_NV}/nccl/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "=== Verifying CUDA visibility ==="
$PY -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "================"

CONCURRENCY=4

echo "=== [1/2] Hallway: fast env ==="
$PY run_grid.py \
    --envs hallway \
    --mixers vdn qmix \
    --explorations independent correlated \
    --similarities obs q_values hidden \
    --seeds 0 1 2 3 4 \
    --episodes 1500 \
    --concurrency $CONCURRENCY \
    --parameter-sharing

echo "=== [2/2] LBF: hard env ==="
$PY run_grid.py \
    --envs lbf \
    --mixers vdn qmix \
    --explorations independent correlated \
    --similarities obs q_values hidden \
    --seeds 0 1 2 3 4 \
    --episodes 3000 \
    --concurrency $CONCURRENCY \
    --parameter-sharing \
    --balanced-buffer

echo "=== All done at $(date) ==="

# Generate plots immediately so they're ready when you log in.
echo "=== Generating plots ==="
$PY plot_results.py --env hallway --mixer vdn --out plot_hallway_vdn.png --window 30 || true
$PY plot_results.py --env hallway --mixer qmix --out plot_hallway_qmix.png --window 30 || true
$PY plot_results.py --env lbf --mixer vdn --out plot_lbf_vdn.png --window 50 || true
$PY plot_results.py --env lbf --mixer qmix --out plot_lbf_qmix.png --window 50 || true

echo "=== Plots saved. Final ls of results: ==="
ls -la results/ | head -30
