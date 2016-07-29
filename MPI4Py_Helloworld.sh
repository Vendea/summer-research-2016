#!/bin/bash

#SBATCH --ntasks=8
#SBATCH -t 00:00:30

. ~/.profile

module load gcc/6.1.0
module load openmpi/1.10.2
module load cuDNN/v4.0
module load cuda/7.5.18
unsetenv PYTHONHOME
. ~/.python/python/bin/activate

mpiexec  python -m mpi4py helloworld