#!/bin/bash


mpirun -np 2 python DistributedMain.py
mpirun -np 3 python DistributedMain.py
mpirun -np 4 python DistributedMain.py
mpirun -np 5 python DistributedMain.py
mpirun -np 6 python DistributedMain.py
mpirun -np 7 python DistributedMain.py
mpirun -np 8 python DistributedMain.py


