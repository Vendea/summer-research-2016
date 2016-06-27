__author__ = 'billywu'

import numpy as np
import tensorflow as tf
from numpy.random import normal
import math
import distributions as dist
from mpi4py import MPI
import sys

class SPSA:
    def __init__(self,cost,feed,var_t,sess):
        self.cost=cost                  # cost function to be minimized
        self.feed=feed                  # feed variables
        self.var_t=[]                   # list of dictionary containing the trainable variables
        self.var=[]                     # this part will convert the variables(tensors) into a list of real numbers(x)
        self.sess=sess
        for tl in var_t:
            for t in tl:
                self.var_t.append(tl[t])
                self.var.append(tl[t].eval(session=sess))

    def set_var(self,var):
        self.var=var
        l=[]
        for v,t in zip(self.var,self.var_t):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))
    def spawn(self,children):
        self.children = children
        self.comm = MPI.COMM_WORLD.Spawn(sys.executable,args=["DSPSA.py"],maxprocs=children)

    def minimize(self,steps,cost,n,c=1,q=0.001,a=0.001,A=100,alpha=0.602,gamma=0.101,limit=1):
        k,shape,data = [],[],[]
        for i in range(self.children):
            for key in feed:
                k.append(key.name)
                shape.append(key.shape)
                data.append(feed[key])
            comm.send(False, dest=i, tag=10)
            comm.send([cost.name,k,shape,tf.get_default_graph().as_graph_def(),
                steps,n,c,q,a,A,alpha,gamma,limit,data], dest=i, tag=11)
        ls = []
        lowest = self.sess.run(cost)
        n1 = minimize(cost,n,c,q,a,A,alpha,gamma,limit)
        for i in range(children):
            ls.append(comm.recv(source=i,tag=11))
        for c,n in ls:
            if c < lowest:
                lowest = c
                n1 = n

        self.set_var(n1)
        
    def kill():
        for i in range(self.children):
            self.comm.send(True, dest=i, tag=10)
    def minimize(self,cost,n,c=1,q=0.001,a=0.001,A=100,alpha=0.602,gamma=0.101,limit=1):
        cn=(c+0.0)/(n+A)**gamma
        an=a/(n+1+A)**alpha
        qk=math.sqrt(q/(n+A)*math.log(math.log(n+A)))
        wk=normal()
        dv=[]
        sess=self.sess
        g=[]
        orig=self.var
        for i in range(limit):
            for m in self.var:
                shape=m.shape
                nm=np.ones(shape=shape)
                for x in np.nditer(nm, op_flags=['readwrite']):
                    x[...]=dist.bernoulli() * 2 * cn
                dv.append(nm)
            l=[]
            for m,d,t in zip(self.var,dv,self.var_t):
                l.append(t.assign(m+d))
            sess.run(tf.group(*l))
            f1=sess.run(cost,self.feed)
            l=[]
            for m,d,t in zip(self.var,dv,self.var_t):
                l.append(t.assign(m-d))
            sess.run(tf.group(*l))
            f0=sess.run(cost,self.feed)
            df=f1-f0
            for m in dv:
                for x in np.nditer(m, op_flags=['readwrite']):
                    x[...]=-(df+0.0)/x/2
            g.append(dv)
        dv=np.average(g,axis=0)
        update=[]
        l=[]
        for m,d,t in zip(self.var,dv,self.var_t):
            e=m+d*an+qk*wk
            update.append(e)
            l.append(t.assign(e))
        sess.run(tf.group(*l))
        self.var=update
        return orig,update

if __name__ == "__main__":
    master = 0 
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    size = comm.Get_size()
    die = False    
    with tf.Session() as sess:
        while not die:
            die = comm.recv(source=master,tag=10)
            cost,feed,shape,g,steps,n,c,q,a,A,alpha,gamma,limit,data = comm.recv(source=master,tag=11)
            i_map = dict()
            temp_vars = []
            temp = dict()
            for n,s,f in zip(feed,shape,data):
                t = tf.Variable(s)
                temp_vars.append(t)
                temp[t] = f
                i_map[n] = t
            feed = temp
            cost = tf.import_graph_def(g,name="",return_elements=cost,input_map=i_map)
            g = tf.get_default_graph()
            var_t = tf.trainable_variables() 
            opt = SPSA(cost,feed,var_t,sess) 
            for x in range(steps):
                orig,update = opt.minimize(cost=cost,n=n,c=c,q=q,a=a,A=A,alpha=alpha,gamma=gamma,limit=limit)
                c = sess.run(cost,feed)
            comm.send([c,update], dest=master, tag=11)


