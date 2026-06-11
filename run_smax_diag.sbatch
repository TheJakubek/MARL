#!/bin/bash
#SBATCH --job-name=smax-diag
#SBATCH --partition=common
#SBATCH --qos=jwozniak_common
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=cluster_logs/smax_diag_%j.out
#SBATCH --error=cluster_logs/smax_diag_%j.err

# Diagnostic: single config with the improved learning setup (8 updates/iter).
# Goal: confirm the win rate actually CLIMBS before committing a full grid.
# Watch the `win=` field in the log — it should rise above ~0.1-0.3.

set -euo pipefail

echo "=== Job info ==="; echo "Node: $(hostname)"; nvidia-smi || true; echo "==="

mkdir -p cluster_logs results_smax_diag

VENV_NV="$(uv run python -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
export LD_LIBRARY_PATH="${VENV_NV}/cudnn/lib:${VENV_NV}/nccl/lib:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8   # single job -> can use most of the card

uv run python -c "import jax; print('jax devices:', jax.devices())"

echo "=== Diagnostic: QMIX + correlated/obs, 2M steps, 8 updates/iter ==="
uv run python -m smax.run \
    --exploration correlated --similarity obs --mixer qmix \
    --seed 0 --total-steps 2000000 --n-envs 128 \
    --updates-per-iter 8 --warmup 5000 \
    --out results_smax_diag/diag_qmix_corr_obs.npz

echo "=== Diagnostic done ==="
