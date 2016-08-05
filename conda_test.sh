#!/bin/bash

#SBATCH --ntasks=1
#SBATCH -t 00:00:30
#SBATCH --gres=gpu
. ~/.profile

module load gcc/6.1.0
module load openmpi/1.10.2
module load cuDNN/v4.0
module load cuda/7.5.18
cd ~/anaconda2/bin/
source activate tensorflow
cd
cd summer-research-2016
mpirun python deep_test.py