#!/bin/bash
#SBATCH --job-name=regression_task
#SBATCH --output=log_slurm/regression_%A_%a.out 
#SBATCH --error=log_slurm/regression_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=20
#SBATCH --time 2:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=margpu[002,003,007,005]
#SBATCH --array=44

## launch script on every task
export LD_LIBRARY_PATH=/home/mind/jrobador/.local/miniconda3/lib
eval "$(conda shell.bash hook)"
conda activate pavi

set -x
time srun python -u script_postprocess.py -n ${SLURM_ARRAY_TASK_ID}
date
