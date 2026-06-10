#!/bin/bash
#SBATCH --job-name=smax-probe
#SBATCH --partition=common
#SBATCH --qos=jwozniak_common
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=cluster_logs/smax_probe_%j.out
#SBATCH --error=cluster_logs/smax_probe_%j.err

# Sanity probe: import jaxmarl, build 2s3z, step once, check GPU.
# Run before any real SMAX training.

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      $(hostname)"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-N/A}"
nvidia-smi || true
echo "================"

mkdir -p cluster_logs

# Same trick as torch: point linker at venv-shipped CUDA libs.
VENV_NV="$(uv run python -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
export LD_LIBRARY_PATH="${VENV_NV}/cudnn/lib:${VENV_NV}/nccl/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

uv run python smax/probe.py
