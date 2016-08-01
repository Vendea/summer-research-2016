import tensorflow as tf
from mpi4py import MPI
import numpy as np 
from sys import path
from os import getcwd

p = getcwd()[0:getcwd().rfind("/")]+"/SGD"
path.append(p)
from ParamServer import ParamServer
from ModelReplica import DPSGD

p = getcwd()[0:getcwd().rfind("/")]+"/lbfgs"
path.append(p)
from lbfgs_optimizer import lbfgs_optimizer
from Opserver import Opserver

p = getcwd()[0:getcwd().rfind("/")]+"/SPSA"
path.append(p)
from SPSA import SPSA

p = getcwd()[0:getcwd().rfind("/")]+"/MCMC"
path.append(p)
from Multi_try_Metropolis import MCMC

p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)
import Logger


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



class DTrainer:
    def __init__(self,cost,x,y,sess=tf.Session(),test,train,batch_size,learning_rate = 0.001,file_name="Error"):
        self._train   = train
        self._test    = test
        self.lr       = learning_rate
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._cost = cost
        self.x = x
        self.y = y 
        self.sess = sess
        self.logfile = Logger.DataLogger(file_name+".csv","core,Epoch,time,train_accuaracy,test_accuaracy,train_cost,test_cost") 
    def write_data(timer,epoch,train_accuaracy,test_accuaracy,train_cost,test_cost):
            if rank == 0:
                for c in range(1,size):
                	comm.recv(source=0,tag=10)
                logfile.writeData((rank,epoch,time.time()-start,train_accuaracy,test_accuaracy,train_cost,test_cost))
            else:
            	comm.send((rank,epoch,time.time()-start,train_accuaracy,test_accuaracy,train_cost,test_cost),dest=0,tag=10)
    def train_step():
    	print("Needs to be implmented")
class DSPSA(DTrainer):
    def __init__(self,num_groups,cost,x,y,sess,test,train,batch_size,learning_rate = 0.001,file_name="Error"):
        self._group   = [x for x in range(size) if x% num_groups == rank%num_groups]
        self._parents = list(set([x% num_groups for x in range(size)]))
        super(DTrainer, self).__init__(cost,x,y,sess,test,train,batch_size,learning_rate,file_name)
        data_x,data_y = self._train
        feed={self.x:data_x,self.y:data_y}
        self.mini=SPSA(self.cost,feed,self.sess)

    def train_step():
        if rank==0:
            orig=self.mini.var
            g=[[self.mini.getGrad(cost,n)]]
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
class DLBFGS(DTrainer):
	def __init__(self,,cost,x,y,sess,test,train,batch_size,learning_rate = 0.001,file_name="Error"):
        super(DTrainer, self).__init__(cost,x,y,sess,test,train,batch_size,learning_rate,file_name)
        data_x,data_y = self._train
        feed={self.x:data_x,self.y:data_y}
        self.mini=SPSA(self.cost,feed,self.sess)
    def train_step():
		if rank==0:
		    trainer=lbfgs_optimizer(self.lr, self.cost,[],self.sess,1,comm,size,rank)   
	        c = trainer.minimize()
		else:
		    opServer=Opserver(self.lr, self.cost,[],self.sess,comm,size,rank,0,self.x,self.y,keep_prob=None)
		    opServer.run()
class DMCMC(DTrainer):
	def __init__(self,,cost,x,y,sess,test,train,batch_size,learning_rate = 0.001,file_name="Error"):
        super(DTrainer, self).__init__(cost,x,y,sess,test,train,batch_size,learning_rate,file_name)
        data_x,data_y = self._train
        feed={self.x:data_x,self.y:data_y}
        self.mini=SPSA(self.cost,feed,self.sess)
    def train_step():
		if rank==0:
		    trainer=lbfgs_optimizer(self.lr, self.cost,[],self.sess,1,comm,size,rank)   
	        c = trainer.minimize()
		else:
		    opServer=Opserver(self.lr, self.cost,[],self.sess,comm,size,rank,0,self.x,self.y,keep_prob=None)
		    opServer.run()
class DSGD(DTrainer):
