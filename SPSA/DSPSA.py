import tensorflow as tf
from mpi4py import MPI
import numpy as np 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



class DTrainer:
    def __init__(self,cost,test,train,batch_size,learning_rate = 0.001):
        self._train   = train
        self._test    = test
        self.lr       = learning_rate
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._cost = cost 

class DSPSA(DTrainer):
    def __init__(self,cost,test,train,batch_size,learning_rate = 0.001,num_groups):
        self._group   = [x for x in range(size) if x% num_groups == rank%num_groups]
        self._parents = list(set([x% num_groups for x in range(size)]))
        super(DTrainer, self).__init__(cost,test,train,batch_size,num_epochs,learning_rate)
        data_x,data_y = train
        feed={x:data_x,y:data_y}
        self.mini=SPSA(cost,feed,sess)

    def train_step():
        if rank==0:
            orig=self.mini.var
            g=[[mini.getGrad(cost,n)]]
            self._group.remove(rank)
            for x in self._group:
                g.append([comm.recv(source=x,tag=11)]) 
            g = np.average(g,axis=0)
            c,update0=self.mini.minimize(g[0],orig,n)
            updates = [(update0,c)]
            for x in parents:
                updates.append(comm.recv(source=x,tag=11))
            c,updates = updates
            update=comm.bcast(updates[c.index(min(c))],root=0)
            self.mini.set_var(update)
        elif rank in parents:
            orig=self.mini.var
            g=[[mini.getGrad(cost,n)]]
            group.remove(rank)
            for x in self._group:
                g.append(comm.recv(source=x,tag=11))
            g = np.average(g,axis=0)
            f,update=self.mini.minimize(g[0],orig,n)
            comm.send((update,f),dest=0,tag=11)
            update=comm.bcast(None,root=0)
            self.mini.set_var(update)
        else :
            g=self.mini.getGrad(cost,n)
            comm.send(g,dest=rank%num_groups,tag=11)
            update=comm.bcast(None,root=0)
            self.mini.set_var(update)
