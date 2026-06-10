#!/bin/bash
#SBATCH --job-name=smax-smoke
#SBATCH --partition=common
#SBATCH --qos=jwozniak_common
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=cluster_logs/smax_smoke_%j.out
#SBATCH --error=cluster_logs/smax_smoke_%j.err

# Smoke test: tiny SMAX run, 5k steps, single config. Should finish in ~5 min.
# Verifies: JAX sees GPU, jaxmarl works, Flax/optax compiles, training loop OK.

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      $(hostname)"
nvidia-smi || true
echo "================"

mkdir -p cluster_logs results_smax_smoke

VENV_NV="$(uv run python -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
export LD_LIBRARY_PATH="${VENV_NV}/cudnn/lib:${VENV_NV}/nccl/lib:${LD_LIBRARY_PATH:-}"

# Tell JAX to use GPU explicitly.
export JAX_PLATFORMS=cuda

uv run python -c "import jax; print('jax devices:', jax.devices()); print('backend:', jax.default_backend())"

echo "=== Smoke: VDN + correlated/obs, 5k steps ==="
uv run python -m smax.run \
    --exploration correlated --similarity obs --mixer vdn \
    --seed 0 --total-steps 5000 --warmup 500 \
    --out results_smax_smoke/smoke_vdn_corr_obs.npz

echo "=== Smoke OK ==="
ls results_smax_smoke/
