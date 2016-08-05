#!/bin/bash

#SBATCH --ntasks=1
#SBATCH -t 00:00:30
#SBATCH --gres=gpu
. ~/.profile

module load gcc/6.1.0
module load openmpi/1.10.2
module load cuDNN/v4.0
module load cuda/7.5.18
source ~/anaconda2/bin/activate

mpirun python deep_test.py