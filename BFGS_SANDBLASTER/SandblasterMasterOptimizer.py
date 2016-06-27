__author__ = 'billywu'


__author__ = 'billywu'

import numpy as np
import tensorflow as tf
import math
import time

rho=0.01
sig=0.5
INT=0.1
ext=3.0
nmax=20
ratio=100
red=1

class BFGSoptimizer:
    def __init__(self,cost,feed,var_t,sess,workers):
        self.cost=cost
        self.feed=feed
        self.var_t=[]
        self.var_v=[]
        self.sess=sess
        self.line_search_fail=False
        for tl in var_t:
            for t in tl:
                self.var_t.append(tl[t])
                self.var_v.append(self.sess.run(tl[t]))
        self.var_v=np.array(self.var_v)
        self.grad_t = tf.gradients(cost,self.var_t)
        self.f1,self.df1=self.ComputeGradient(self.var_v)
        self.s=-self.df1
        self.d1=-self.var_self_inner(self.s)
        self.z1=red/(1-self.d1)
        self.workers=workers



    def ComputeGradient(self,var_v):
        self.update()
        start=time.time()
        c=self.sess.run(self.cost,self.feed)
        grad=np.array(self.sess.run(self.grad_t, feed_dict=self.feed))
        end=time.time()
        print "gradient computation",end-start
        return c,grad

    def var_self_inner(self,var_v1):
        v=[]
        for m in var_v1:
            v=v+[x for x in np.nditer(m, op_flags=['readwrite'])]
        v=np.array(v)
        s=np.inner(v,v)
        return s

    def var_inner(self,var_v1,var_v2):
        v1=[]
        v2=[]
        for m1,m2 in zip(var_v1,var_v2):
            v1=v1+[x for x in np.nditer(m1, op_flags=['readwrite'])]
            v2=v2+[x for x in np.nditer(m2, op_flags=['readwrite'])]
        return np.inner(v1,v2)

    def update(self):
        l=[]
        for t,v in zip(self.var_t,self.var_v):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))

    def update_feed(self,feed):
        self.feed=feed


    def minimize(self,alpha=0.0001):
        x0=self.var_v; f0=self.f1;df0=self.df1
        self.var_v=self.var_v+self.z1*self.s
        f2,df2=self.ComputeGradient(self.var_v)
        d2=self.var_inner(df2,self.s)
        f3=self.f1;d3=self.d1;z3=-self.z1
        M=20; success=0;limit=-1
        while True:
            while ((f2>self.f1+self.z1*rho*self.d1) or
                       (d2>-sig*self.d1)) and (M>0):
                limit=self.z1
                try:
                    if f2>self.f1:
                        z2=z3-(0.5*d3*z3*z3)/(d3*z3+f2-f3)
                    else:
                        A=6*(f2-f3)/z3+3*(d2+d3)
                        B = 3*(f3-f2)-z3*(d3+2*d2)
                        z2 = (math.sqrt(B*B-A*d2*z3*z3)-B)/A
                except:
                    z2=z3/2
                if np.isinf(z2) or np.isnan(z2):
                    z2=z3/2
                z2 = max(min(z2, INT*z3),(1-INT)*z3)
                self.z1=self.z1+z2
                self.var_v=self.var_v+z2*self.s
                f2,df2=self.ComputeGradient(self.var_v)
                M=M-1
                d2=self.var_inner(df2,self.s)
                z3=z3-z2
            if f2 > self.f1+self.z1*rho*self.d1 or d2 > -sig*self.d1:
                break #fail
            elif d2>sig*self.d1:
                success=1
                break
            elif M==0:
                break
            A = 6*(f2-f3)/z3+3*(d2+d3)
            B = 3*(f3-f2)-z3*(d3+2*d2)
            try:
                z2 = -d2*z3*z3/(B+math.sqrt(B*B-A*d2*z3*z3))
                if ~np.isreal(z2) or np.isnan(z2) or np.isinf(z2) or z2 < 0:
                    if limit<-0.5:
                        z2 = self.z1 * (ext-1)
                    else:
                        z2 = (limit-self.z1)/2
                elif (limit > -0.5) & (z2+self.z1 > limit):
                    z2 = (limit-self.z1)/2
                elif (limit < -0.5) & (z2+self.z1 > self.z1*ext):
                    z2 = self.z1*(ext-1.0)
                elif z2 < -z3*INT:
                    z2 = -z3*INT
                elif (limit > -0.5) & (z2 < (limit-self.z1)*(1.0-INT)):
                    z2 = (limit-self.z1)*(1.0-INT)
            except:
                if limit<-0.5:
                    z2 = self.z1 * (ext-1)
                else:
                    z2 = (limit-self.z1)/2
            f3=f2;d3=d2;z3=-z2
            self.z1=self.z1+z2
            self.var_v=self.var_v+z2*self.s
            f2,df2=self.ComputeGradient(self.var_v)
            M=M-1
            d2=self.var_inner(df2,self.s)
        if success==1:
            self.f1 = f2
            self.s = (self.var_inner(df2,df2)-self.var_inner(self.df1,df2))/(self.var_self_inner(self.df1))*self.s - df2
            tmp=self.df1; self.df1=df2; df2=tmp
            d2=self.var_inner(self.df1,self.s)
            if d2>0:
                self.s=-self.df1
                d2=-self.var_self_inner(self.s)
            self.z1=self.z1*min(ratio,self.d1/(d2-0.000000000001))
            self.d1 = d2
            self.line_search_fail=False
        else:
            self.var_v = x0; self.f1 = f0; df1 = df0
            if self.line_search_fail:
                self.var_v=self.var_v+df1*alpha
                self.update()
                return
            tmp = df1; self.df1 = df2; df2 = tmp
            self.s = -self.df1
            self.d1 = -self.var_self_inner(self.s)
            self.z1 = 1/(1-self.d1)
            self.ls_failed = True
        self.update()