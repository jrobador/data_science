#!/bin/bash
#SBATCH --job-name=regression_task
#SBATCH --output=log_slurm/regression_%A_%a.out 
#SBATCH --error=log_slurm/regression_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --time 12:00:00
#SBATCH --partition=gpu-best

## launch script on every task
export LD_LIBRARY_PATH=/home/mind/jrobador/.local/miniconda3/lib
eval "$(conda shell.bash hook)"
conda activate pavi

set -x
time srun python -u resnet50_regression.py
date
