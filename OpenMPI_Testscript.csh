#!/bin/bash
#SBATCH -t 1
#SBATCH -n 4
#SBATCH --mem-per-cpu=128
#SBATCH --share

. ~/.profile

module load python/2.7.8
module load openmpi/1.6.5

mpiexec  python -m mpi4py helloworld
hostname
date