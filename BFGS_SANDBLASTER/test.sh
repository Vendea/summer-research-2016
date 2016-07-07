#!/bin/bash


mpirun -hostfile hostfile -np 2 python DistributedMain.py 2
mpirun -hostfile hostfile -np 3 python DistributedMain.py 3
mpirun -hostfile hostfile -np 4 python DistributedMain.py 4
mpirun -hostfile hostfile -np 5 python DistributedMain.py 5
mpirun -hostfile hostfile -np 6 python DistributedMain.py 6
mpirun -hostfile hostfile -np 7 python DistributedMain.py 7
mpirun -hostfile hostfile -np 8 python DistributedMain.py 8


