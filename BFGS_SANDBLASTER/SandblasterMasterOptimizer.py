__author__ = 'billywu'

import math
import time

import numpy as np
import tensorflow as tf


#Global parameters

rho=0.01
sig=0.5
INT=0.1
ext=3.0
nmax=20
ratio=100
red=1

class BFGSoptimizer:

    ''' Initializing the Optimizer Object
        Param:
            grad_compute_time_accumulator:              how much time is spent so far on computing the gradient
            grad_counter:                               how many times the gradient is so far computed
            cost:                                       cost/loss/objective function to be minimized
            feed:                                       what feed to use when getting the function value
            rank:                                       On which core this object is initialized
            x_dir:                                      The shared location for the parameters
            comm:                                       The MPI communicator
            size:                                       Total MPI processes number
            var_t:                                      parameters tensors
            var_v:                                      parameters values
            grad_t:                                     gradient tensor
            f1,df1,s,z1:                                lbfgs variables
    '''
    def __init__(self,cost,feed,sess,rank, comm):
        self.grad_compute_time_accumulator=0
        self.grad_counter=0
        self.cost=cost
        self.feed=feed
        self.rank=rank
        self.comm=comm
        self.size=comm.Get_size()
        self.var_t=tf.trainable_variables()
        self.var_v=[]
        vv=[]
        self.sess=sess
        self.line_search_fail=False
        for tl in self.var_t:
            vv.append(self.sess.run(tl))
        self.var_v=np.array(vv)
        self.grad_t = tf.gradients(cost,self.var_t)
        self.f1,self.df1=self.ComputeGradient(self.var_v)
        self.s=-self.df1
        self.d1=-self.var_self_inner(self.s)
        self.z1=red/(1-self.d1)

    # Distributed Computation of the Gradient, offering a 15% speedup than tensorflow's distributed method when
    # using the same resources

    def ComputeGradient(self,var_v):
        start=time.time()
        self.comm.scatter([self.var_v for x in range(self.size)],root=self.rank)
        g=self.Compute_Gradient(self.var_v)
        c=self.Compute_Cost()
        new_data = self.comm.gather((g,c),root=0)
        gr=new_data[0][0]
        ct=new_data[0][1]
        for i in range(1,len(new_data)):
            gr=gr+new_data[i][0]
            ct=ct+new_data[i][1]
        end=time.time()
        #print "Cost function:", ct
        #print "Computation Time:", end-start
        self.grad_compute_time_accumulator=self.grad_compute_time_accumulator+end-start
        self.grad_counter=self.grad_counter+1
        return ct,gr

    # reporting gradient computational time
    def get_average_grad_time(self):
        return self.grad_compute_time_accumulator/self.grad_counter

    # Compute the partial gradient on the main core
    def Compute_Gradient(self,var):
        self.Assign_Gradient(var)
        g=np.array(self.sess.run(self.grad_t,self.feed))
        return g

    # Compute the partial cost function on the main core
    def Compute_Cost(self):
        c=np.array(self.sess.run(self.cost,self.feed))
        return c

    # Apply the parameters stored in the shared location
    def Assign_Gradient(self, var):
        self.x=var
        l=[]
        for t,v in zip(self.var_t,self.x):
            l.append(t.assign(v))
        self.sess.run(tf.group(*l))

    # Numpy unstructured inner product of a vector and itself
    def var_self_inner(self,var_v1):
        v=[]
        for m in var_v1:
            v=v+[x for x in np.nditer(m, op_flags=['readwrite'])]
        v=np.array(v)
        s=np.inner(v,v)
        return s

    # Numpy unstructured inner product
    def var_inner(self,var_v1,var_v2):
        v1=[]
        v2=[]
        for m1,m2 in zip(var_v1,var_v2):
            v1=v1+[x for x in np.nditer(m1, op_flags=['readwrite'])]
            v2=v2+[x for x in np.nditer(m2, op_flags=['readwrite'])]
        return np.inner(v1,v2)

    # update the feed
    def update_feed(self,feed):
        self.feed=feed

    # stand alone lbfgs inspired by mark schmidt's minifunc, using a linesearch as well as a learning rate
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
                return
            tmp = df1; self.df1 = df2; df2 = tmp
            self.s = -self.df1
            self.d1 = -self.var_self_inner(self.s)
            self.z1 = 1/(1-self.d1)
            self.ls_failed = True