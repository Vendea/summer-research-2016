__author__ = 'billywu'

import tensorflow as tf
import numpy as np


class Opserver:
    def __init__(self,learning_rate, cost,feed,sess,comm,size,rank,root):
        self.cost=cost
        self.feed=feed
        self.sess=sess
        self.rank=rank
        self.comm=comm
        self.root=root
        self.size=size
        self.gradient=tf.gradients(cost,tf.trainable_variables())
        v=[]
        for t in tf.trainable_variables():
            v.append(sess.run(t))
        self.var=np.array(v)
        self.learningRate=learning_rate
        self.old_grad=None
    def update_var(self,var=None):
        l=[]
        if var==None:
            var=self.var
        for v,t in zip(var,tf.trainable_variables()):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))

    def run(self):
        while (True):
            data=self.comm.scatter(['None' for x in range(self.size)],root=self.root)
            if data[0]=="G":
                self.update_var(data[1])
                g=np.array(self.sess.run(self.gradient,self.feed))
                self.comm.gather(g,root=self.root)
            elif data[0]=="C":
                self.update_var(data[1])
                c=self.sess.run(self.cost,self.feed)
                self.comm.gather(c,root=self.root)
            elif data[0]=="K":
                break
        print "Core,", self.rank, "Finish"

