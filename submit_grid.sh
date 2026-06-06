#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --partition=common
#SBATCH --qos=jwozniak_common
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --output=grid_output_%j.txt

/home/jwozniak/.local/bin/uv run run_grid.py
