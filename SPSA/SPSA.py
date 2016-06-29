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
        for tl in self.var_t:
            self.var.append(tl.eval(session=sess))

    def set_var(self,var):
        self.var=var
        l=[]
        for v,t in zip(self.var,self.var_t):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))

    def minimize(self,cost,n,c=1,q=0.0001,a=0.0001,A=100,alpha=0.602,gamma=0.101,limit=1):
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




