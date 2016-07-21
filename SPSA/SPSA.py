__author__ = 'billywu'

import numpy as np
import tensorflow as tf
from numpy.random import normal
import math
import distributions as dist

class SPSA:
    def __init__(self,cost,feed,sess):
        self.cost=cost                  # cost function to be minimized
        self.feed=feed                  # feed variables
        self.var_t=tf.trainable_variables()                   # list of dictionary containing the trainable variables
        self.var=[]                     # this part will convert the variables(tensors) into a list of real numbers(x)
        self.sess=sess
        self.assign_placeholders=[]
        assign_op=[]
        for t in self.var_t:
            self.var.append(t.eval(session=sess))
            self.assign_placeholders.append(tf.placeholder(shape=self.var[-1].shape,dtype="float32"))
            assign_op.append(t.assign(self.assign_placeholders[-1]))
        self.assign=tf.group(*assign_op)

    def set_var(self,var):
        self.var=var
        feed={}
        for t,v in zip(self.assign_placeholders,var):
            feed[t]=v
        self.sess.run(self.assign,feed)

    def getGrad(self,cost,n,c=1,q=0.0001,a=0.0001,A=100,alpha=0.602,gamma=0.101):
        cn=(c+0.0)/(n+A)**gamma
        an=a/(n+1+A)**alpha
        qk=math.sqrt(q/(n+A)*math.log(math.log(n+A)))
        wk=normal()
        dv=[]
        dv1=[]
        dv2=[]
        for m in self.var:
            shape=m.shape
            nm=np.ones(shape=shape)
            for x in np.nditer(nm, op_flags=['readwrite']):
                x[...]=dist.bernoulli() * 2 * cn
            dv.append(nm)
            dv1.append(nm+m)
            dv2.append(nm-m)
        self.set_var(dv1)
        f1=self.sess.run(cost,self.feed)
        self.set_var(dv2)
        f2=self.sess.run(cost,self.feed)
        g=[]
        for m in dv:
            nm=(f1-f2)/2/m
            g.append(nm)
        return g


    def minimize(self,g,orig,n,c=1,q=0.01,a=0.001,A=100,alpha=0.602,gamma=0.101,limit=1):
        an=a/(n+1+A)**alpha
        qk=math.sqrt(q/(n+A)*math.log(math.log(n+A)))
        wk=normal()
        dv=[]
        sess=self.sess
        update=[]
        for m,gr in zip(orig,g):
            update.append(m-gr*an)
        self.set_var(update)
        return self.sess.run(self.cost,self.feed),update






