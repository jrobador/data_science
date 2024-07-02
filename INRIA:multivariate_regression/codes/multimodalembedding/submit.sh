#!/bin/bash
#SBATCH --job-name=multivariate_reg
#SBATCH --output=log_slurm/multivariate_reg%A.out 
#SBATCH --error=log_slurm/multivariate_reg%A.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=30
#SBATCH --time 12:00:00
#SBATCH --partition=gpu-best

plots_dir="experiments/plots_$SLURM_JOB_ID"
mkdir -p $plots_dir

## launch script on every task
export LD_LIBRARY_PATH=/home/mind/jrobador/.local/miniconda3/lib
eval "$(conda shell.bash hook)"
conda activate pavi


set -x
time srun python -u multivariate_regression.py $plots_dir
date
