#!/bin/bash
#SBATCH --job-name=aemps_3x8           # Job name
#SBATCH --output=logs/%x_%j.out       # Standard output (%x-job name, %j-job ID)
#SBATCH --error=logs/%x_%j.err        # Error log
#SBATCH --partition=cpu               # Partition (queue) name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=128             # Number of CPUs per task
#SBATCH --time=24:00:00               # Time limit hrs:min:sec

working_dir=/home/keisuke/QC_MPS
cd $working_dir

source ~/miniconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate QC_MPS

which python
echo "Current shell: $SHELL"

# Base seed can be defined here.
BASE_SEED=1234

python -u train_mnist.py \
    --epochs 100 \
    --seed 2024 \
    --lr 0.0001 \
    mpsae \
    --total_schedule_steps 10 \
    --mode adaptive \
    --manifold Exact \
    --conv_strategy relative \
    --conv_threshold 1e-2 \
    --min_epochs 90 \
    --simple_epochs 10 \
    --simple_lr 0.00005 --schedule_type sr_base > logs/train_mnist_${SLURM_JOB_ID}_output.log 2>&1


# python -u train_mnist.py \
#     --epochs 50 \
#     --seed $BASE_SEED \
#     --lr 0.0001 \
#     mpsae \
#     --mode plain \
#     --manifold Original \
#     --conv_strategy relative \
#     --conv_threshold 1e-3 \
#     --simple_epochs 10 \
#     --simple_lr 0.0001 > logs/train_mnist_${SLURM_JOB_ID}_output.log 2>&1