#!/bin/bash

for i in `seq 15000 100000`
do
    if [ $((i%5000 == 0)) -eq 1 ]
    then
        echo "start training size: $i"
        mpirun -np 1 python2 generalneuralnets3.py $i True
        mpirun -np 4 python2 generalneuralnets3.py $i True
        mpirun -np 1 python2 generalneuralnets3.py $i False
        mpirun -np 4 python2 generalneuralnets3.py $i False
        echo "end training size: $i"
    fi
done
