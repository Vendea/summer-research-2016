import tensorflow as tf
import numpy as np
from mpi4py import MPI
import sys,ast
'''
   ===============================================================================================
    Command Line Arguments 
   ===============================================================================================
        Optimizer 1
        Number of workers 2
        size of the batchs if zero no batching should occur 3
        cost function 4
        activation function 5
        accuarcy checking method 6
        number of elemetns in train set 7
        number of elements in test set 8
        trainning epochs 9
        network [layers,nodes] 10
        loc of data [train,test] 11
        State key either MASTER or SLAVE 12
        rank passed to master only 13
'''
args = sys.argv[1:]
temp =[]
for x in args:
	temp.append(eval(x))

         
args = temp
opt           = [str(x) for x in args[0]]
workers       = [str(x) for x in args[1]]
batch_size    = [str(x) for x in args[2]]
cost          = [str(x) for x in args[3]]
kernal        = [str(x) for x in args[4]]
accuarcy      = [str(x) for x in args[5]]
train         = [str(x) for x in args[6]]
test          = [str(x) for x in args[7]]
epochs        = [str(x) for x in args[8]]
nodes         = [str(x) for x in args[9]]
data          = [str(x) for x in args[10]]
n_workers     = len(opt)
for i in range(n_workers):
	args = [
	    "\""+opt[i]+"\"",
	    workers[i],
	    batch_size[i],
	    cost[i],
	    kernal[i],
	    accuarcy[i],
	    train[i],
	    test[i],
	    epochs[i],
	    nodes[i],
	    data[i],
	    "\"MASTER\"",
	    str(i)]
	comm = MPI.COMM_WORLD.Spawn(sys.executable,args=["GenNet_MNIST.py"]+args,maxprocs=1)