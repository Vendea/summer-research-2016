__author__ = 'billywu'

import numpy as np
from random import random
from numpy.random import normal
import math

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

    def minimize(self,cost,n,c=1,q=0.00001,a=0.01,A=100,alpha=0.602,gamma=0.101,limit=10):
        cn=(c+0.0)/(n+A)**gamma
        an=a/(n+1+A)**alpha
        qk=math.sqrt(q/(n+A)*math.log(math.log(n+A)))
        wk=normal()
        dv=[]
        sess=self.sess
        g=[]
        for i in range(limit):
            for m in self.var:
                shape=m.shape
                nm=np.ones(shape=shape)
                for x in np.nditer(nm, op_flags=['readwrite']):
                    x[...]=(int(random() * 2) - 0.5) * 2 * cn
                dv.append(nm)
            for m,d,t in zip(self.var,dv,self.var_t):
                sess.run(t.assign(m+d))
            f1=sess.run(cost,self.feed)
            for m,d,t in zip(self.var,dv,self.var_t):
                sess.run(t.assign(m-d))
            f0=sess.run(cost,self.feed)
            df=f1-f0
            for m in dv:
                for x in np.nditer(m, op_flags=['readwrite']):
                    x[...]=-(df+0.0)/x/2*an
            g.append(dv)
        dv=np.average(g,axis=0)
        update=[]
        for m,d,t in zip(self.var,dv,self.var_t):
            e=m+d+qk*wk
            update.append(e)
            sess.run(t.assign(e))
        self.var=update




