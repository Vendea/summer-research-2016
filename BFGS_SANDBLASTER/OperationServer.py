__author__ = 'billywu'


import numpy as np
import tensorflow as tf

class SandblasterOpServer:
    def __init__(self, rank, feed, sess, cost, comm):
        self.rank=rank
        self.sess=sess
        self.feed=feed
        self.var_t=tf.trainable_variables()
        self.var_v=[]
        vv=[]
        self.sess=sess
        self.line_search_fail=False
        for tl in self.var_t:
            vv.append(self.sess.run(tl))
        self.var_v=np.array(vv)
        self.cost=cost
        self.grad_t=tf.gradients(cost,self.var_t)

    def Compute_Gradient(self,var_v):
        self.Assign_Gradient(var_v)
        g=np.array(self.sess.run(self.grad_t,self.feed))
        return g

    def Compute_Cost(self):
        c=np.array(self.sess.run(self.cost,self.feed))
        return c

    def Assign_Gradient(self, var_v):
        self.x=var_v
        l=[]
        for t,v in zip(self.var_t,self.x):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))