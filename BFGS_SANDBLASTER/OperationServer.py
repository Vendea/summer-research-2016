__author__ = 'billywu'

from mpi4py import MPI
import numpy as np
import tensorflow as tf

class SandblasterOpServer:
    def __init__(self, rank, x_dir, feed, sess, cost, var_t,comm):
        self.rank=rank
        self.sess=sess
        self.feed=feed
        vv=[]
        self.var_t=[]
        for tl in var_t:
            for t in tl:
                self.var_t.append(tl[t])
                vv.append(self.sess.run(tl[t]))
        self.var_v=np.array(vv)
        try:
            self.x=np.load(x_dir)
        except:
            np.save(x_dir,self.var_v)
        self.x_dir=x_dir
        self.cost=cost
        self.grad_t=tf.gradients(cost,self.var_t)

    def Compute_Gradient(self):
        self.Assign_Gradient(self.x_dir)
        g=np.array(self.sess.run(self.grad_t,self.feed))
        return g

    def Compute_Cost(self):
        c=np.array(self.sess.run(self.cost,self.feed))
        return c

    def Assign_Gradient(self, x_dir):
        self.x=np.load("xdat.npy")
        l=[]
        for t,v in zip(self.var_t,self.x):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))