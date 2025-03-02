#!/bin/bash
#SBATCH --job-name=umps_bp           # Job name
#SBATCH --output=logs/%x_%j.out       # Standard output (%x-job name, %j-job ID)
#SBATCH --error=logs/%x_%j.err        # Error log
#SBATCH --partition=cpu               # Partition (queue) name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=10             # Number of CPUs per task
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --array=1-10                  # Array: run 10 jobs in parallel

working_dir=/home/keisuke/QC_MPS
cd $working_dir

source ~/miniconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate QC_MPS

which python
echo "Current shell: $SHELL"

# Base seed can be defined here.
BASE_SEED=1234

# Loop 10 times per job.
# Compute a unique seed for each run as: BASE_SEED + (JOB_ID - 1)*10 + loop_counter.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    JOB_ID=1
else
    JOB_ID=$SLURM_ARRAY_TASK_ID
fi

for i in {1..10}; do
    SEED=$(( BASE_SEED + (JOB_ID - 1)*10 + i ))
    echo "Running training run with seed ${SEED} (Job ID ${JOB_ID}, iteration ${i})"
    python -u check_umps_bp.py --seed ${SEED} >> logs/umps_bp_${JOB_ID}_${i}.log 2>&1
done
