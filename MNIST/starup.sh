#!/bin/bash

mpirun -np 3 python dsgd.py 
mpirun -np 3 python dlbfg.py 
mpirun -np 3 python dspsa.py 
mpirun -np 3 python dmcmc.py 



