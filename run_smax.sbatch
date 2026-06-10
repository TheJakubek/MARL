#!/bin/bash
#SBATCH --job-name=smax-2s3z
#SBATCH --partition=common
#SBATCH --qos=jwozniak_common
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cluster_logs/smax_%j.out
#SBATCH --error=cluster_logs/smax_%j.err

# ----------------------------------------------------------------------------
# SMAX 2s3z full grid:
#   - mixers:        vdn, qmix
#   - explorations:  independent + correlated × {obs, q_values, hidden}
#   - seeds:         0..4
#   - total_steps:   2M env-steps per run (128 parallel envs via vmap)
#
# 2 mixers × 4 exploration variants × 5 seeds = 40 jobs.
# Concurrency=4 (single GPU shared) -> ~10 sequential waves.
# Throughput depends on GPU; the smoke job reports env-steps/sec — size from that.
# ----------------------------------------------------------------------------

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
nvidia-smi || true
echo "================"

mkdir -p cluster_logs results_smax logs_smax

VENV_NV="$(uv run python -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
export LD_LIBRARY_PATH="${VENV_NV}/cudnn/lib:${VENV_NV}/nccl/lib:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda

# CRITICAL for multiple JAX processes on one GPU: JAX otherwise preallocates
# ~75% of GPU memory per process, so concurrent jobs OOM-crash instantly.
# Hard-cap each process to 40% of the 11GB card -> 2 fit side by side (2*0.4<1).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

uv run python -c "import jax; print('jax devices:', jax.devices()); print('backend:', jax.default_backend())"

echo "=== SMAX 2s3z grid ==="
uv run python run_smax_grid.py \
    --mixers vdn qmix \
    --explorations independent correlated \
    --similarities obs q_values hidden \
    --seeds 0 1 2 3 4 \
    --total-steps 2000000 \
    --concurrency 2

echo "=== All done at $(date) ==="
ls -la results_smax/ | head -50
