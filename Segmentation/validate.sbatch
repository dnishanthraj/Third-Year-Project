#!/bin/bash
#
#SBATCH --job-name=val1 # Job name for tracking
#SBATCH --partition=falcon  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA
python validate.py --name NestedUNET --augmentation True --folder NestedUNET_with_augmentation_20250426_122016_315f9c91

# torchrun --nproc_per_node=NUM_GPUS validate.py ---name NestedUNET --augmentation True --folder NestedUNET_with_augmentation_20250126_131118_4f8e5884
