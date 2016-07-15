__author__ = 'billywu'

import tensorflow as tf
import numpy as np
import time


class Opserver:
    def __init__(self,learning_rate, cost,feed,sess,comm,size,rank,root,x,y,keep_prob):
        self.cost=cost
        self.feed=feed
        self.sess=sess
        self.rank=rank
        self.comm=comm
        self.root=root
        self.size=size
        v=[]
        self.assign_placeholders=[]
        assign_op=[]
        for t in tf.trainable_variables():
            v.append(sess.run(t))
            self.assign_placeholders.append(tf.placeholder(shape=v[-1].shape,dtype="float32"))
            assign_op.append(t.assign(self.assign_placeholders[-1]))
        self.assign=tf.group(*assign_op)
        self.gradient=tf.gradients(cost,tf.trainable_variables())
        comm.scatter(['Init' for i in range(size)],root=root)
        self.var=np.load('var.npy')
        self.learningRate=learning_rate
        self.old_grad=None
        self.x=x
        self.y=y
        self.keep_prob=keep_prob

    def update_var(self,var=None):
        var=np.load('var.npy')
        feed={}
        for t,v in zip(self.assign_placeholders,var):
            feed[t]=v
        self.sess.run(self.assign,feed)

    def run(self):
        while (True):
            data=self.comm.scatter(['None' for x in range(self.size)],root=self.root)
            if data[0]=="G":
                s=time.time()
                g=np.array(self.sess.run(self.gradient,self.feed))
                e=time.time()
                #print "Gradient Server Compute",e-s
                self.comm.gather(g,root=self.root)
            elif data[0]=="C":
                c=self.sess.run(self.cost,self.feed)
                self.comm.gather(c,root=self.root)
            elif data[0]=="K":
                break
            elif data[0]=="U":
                data_x,data_y=data[1]
                self.feed={self.x:data_x,self.y:data_y,self.keep_prob:1.0}
            elif data[0]=="W":
                s=time.time()
                self.update_var()
                e=time.time()
                #print "Update Time", e-s
            elif data[0]=="I":
                datax,datay=data[1]
                ret=np.inner(datax,datay)
                self.comm.gather(ret,root=self.root)
            elif data[0]=="IS":
                datax=data[1]
                ret=np.inner(datax,datax)
                self.comm.gather(ret,root=self.root)
        print "Core,", self.rank, "Finish"

