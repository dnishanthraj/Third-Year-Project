#!/bin/bash
#
#SBATCH --job-name=lidc-seg # Job name for tracking
#SBATCH --partition=gecko  # Partition you wish to use (see above for list)
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
## python3.11 train.py --name UNET --augmentation True ##
# python3.11 validate.py --name UNET --augmentation True
python validate.py --name NestedUNET --augmentation True --folder NestedUNET_with_augmentation_20250124_153410_92030151
