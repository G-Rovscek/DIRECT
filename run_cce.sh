#!/bin/sh
#SBATCH --job-name=cce_direct           # the title of the job
#SBATCH --output=direct_cce.log         # file to which logs are saved
#SBATCH --time=24:00:00                   # job time limit - full format is D-H:M:S
#SBATCH --nodes=1                         # number of nodes
#SBATCH --gres=gpu:1                      # number of gpus
#SBATCH --ntasks=1                        # number of tasks
#SBATCH --mem-per-gpu=82G                 # memory allocation
#SBATCH --partition=gpu                   # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12                # number of allocated cores  

# Activate conda env
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/direct_env

CONFIG="./configs/cce_direct.yaml"

# Run code
srun --nodes=1 --exclusive --gres=gpu:1 --ntasks=1 python3 -m src.experiments.cce \
    --test_config=$CONFIG \
