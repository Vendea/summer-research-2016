__author__ = 'billywu'

import tensorflow as tf
import numpy as np
import math
import time

class lbfgs_optimizer:
    def __init__(self,learning_rate, cost,feed,sess,m,comm,size,rank):
        self.Y=[]
        self.S=[]
        self.YS=[]
        self.cost=cost
        self.sess=sess
        self.NumIter=0
        self.m=m
        self.gradientEval=0
        self.functionEval=0
        self.innerEval=0
        self.HessianEval=0
        self.last_z1=learning_rate
        self.memorySize=0
        self.rank=rank
        self.comm=comm
        self.size=size
        v=[]
        self.assign_placeholders=[]
        assign_op=[]
        for t in tf.trainable_variables():
            v.append(sess.run(t))
            self.assign_placeholders.append(tf.placeholder(shape=v[-1].shape,dtype="float32"))
            assign_op.append(t.assign(self.assign_placeholders[-1]))
        self.assign=tf.group(*assign_op)
        self.var=np.array(v)
        # self.var=np.load('var.npy')
        np.save('var.npy',self.var)
        comm.scatter(['Init' for i in range(size)],root=rank)
        self.gradient=tf.gradients(cost,tf.trainable_variables())
        self.learningRate=learning_rate
        self.old_grad=None

    def update(self,data_x,data_y,x,y,keep_prob=None):
        start=time.time()
        feed=[]
        s=len(data_x)/self.size
        if keep_prob!=None:
            kp=True
        else:
            kp=False
        for i in range(self.size):
            feed.append(("U",(data_x[i*s:(i+1)*s],data_y[i*s:(i+1)*s],kp)))
        data=self.comm.scatter(feed,root=self.rank)
        data_x,data_y,kp=data[1]
        if kp:
            self.feed={x:data_x,y:data_y,keep_prob:1.0}
        else:
            self.feed={x:data_x,y:data_y}
        print "Update Batch:", time.time()-start



    def update_var(self,var=None):
        s=time.time()
        if var==None:
            var=self.var
        feed={}
        for t,v in zip(self.assign_placeholders,var):
            feed[t]=v
        self.sess.run(self.assign,feed)
        np.save('var',var)
        self.comm.scatter([("W",0) for i in range(self.size)],root=self.rank)
        #print "Update Var:", time.time()-s

    def var_self_inner(self,var_v1,useFlatten=False):
        start=time.time()
        self.innerEval=self.innerEval+1
        ret=0
        for m in var_v1:
            v=np.ravel(m)
            ret=ret+np.inner(v,v)
        return ret

    # Numpy unstructured inner product
    def var_inner(self,var_v1,var_v2, useFlatten=False):
        self.innerEval=self.innerEval+1
        s=time.time()
        ret=0
        for m1,m2 in zip(var_v1,var_v2):
             ret=ret+np.inner(np.ravel(m1),np.ravel(m2))
        e=time.time()
        #print "Inner product:", e-s
        return ret



    def update_hessian(self,g,Hdiag):
        self.HessianEval=self.HessianEval+1
        s=time.time()
        d=-g
        d=np.array(d)
        al=np.zeros(self.memorySize)
        be=np.zeros(self.memorySize)
        for j in range(self.memorySize):
            i=self.memorySize-j-1
            al[i]=(self.var_inner(self.S[i],d))/self.YS[i]
            d=d-al[i]*self.Y[i]
        d=Hdiag*d
        for i in range(self.memorySize):
            be[i]=(self.var_inner(self.Y[i],d))/self.YS[i]
            d=d+self.S[i]*(al[i]-be[i])
        #print "Hessian Update:", time.time()-s
        return d



    def poly(self,z0,z1,f0,f1,g0,g1,lower,upper):
        d1 = g0 + g1 - 3*(f0-f1)/(z0-z1)
        if d1*d1 - g1*g0>0:
            d2 = math.sqrt(d1*d1 - g1*g0)
            minPos = z1 - (z1 - z0)*((g1 + d2 - d1)/(g1 - g0 + 2*d2))
            return max(lower,min(upper,minPos))
        else:
            return (lower+upper)/2


    def getGradient(self,var):
        self.gradientEval=self.gradientEval+1
        s=time.time()
        self.update_var(var)
        self.comm.scatter([("G","var") for i in range(self.size)],root=self.rank)
        data=np.array(self.sess.run(self.gradient,self.feed))
        #s=time.time()
        gradients=self.comm.gather(data,root=self.rank)
        ret=gradients[0]
        #s=time.time()
        for i in range(1,len(gradients)):
            ret=np.add(ret,gradients[i])
        e=time.time()
        print "Gradient Time:",e-s
        return ret/self.size

    def kill(self):
        self.comm.scatter([("K",1) for i in range(self.size)],root=self.rank)

    def getFunction(self,var):
        self.functionEval=self.functionEval+1
        s=time.time()
        self.update_var(var)
        self.comm.scatter([("C","var") for i in range(self.size)],root=self.rank)
        data=self.sess.run(self.cost,self.feed)
        costs=self.comm.gather(data,root=self.rank)
        ret=costs[0]
        for i in range(1,len(costs)):
            ret=np.add(ret,costs[i])
        e=time.time()
        #print "Function Time", e-s
        return ret/self.size


    def minimize(self,ls=True):
        if self.NumIter<self.m:
            if self.old_grad==None:
                self.old_grad=self.getGradient(self.var)
            else:
                self.old_grad=self.old_grad
            self.var=self.var-0.00001*self.old_grad
            grad=self.getGradient(self.var)
            s=-0.00001*grad
            y=grad-self.old_grad
            self.S.append(s)
            self.Y.append(y)
            self.YS.append(self.var_inner(self.S[-1],self.Y[-1]))
            self.old_grad=grad
            self.memorySize=self.memorySize+1
            self.NumIter=self.NumIter+1
            return 0,0
        else:
            r=self.update_hessian(self.old_grad,1)
            if ls:
                f0=self.getFunction(self.var)
                g0=self.old_grad
                gtd0=self.var_inner(self.old_grad,r)
                z1=self.last_z1*2
                f1=self.getFunction(self.var+z1*r)
                g1=self.getGradient(self.var+z1*r)
                gtd1=self.var_inner(g1,r)
                lsIter=0
                lsIterMax=10
                f_prev=f0
                g_prev=self.old_grad
                z_prev=0
                gtd_prev=gtd0
                done=False
                while lsIter<lsIterMax:
                    if np.isnan(f1):
                        z1=z1/2.0
                    else:
                        if f1>f0+0.001*z1*gtd0 or (lsIter>1 and f1>f_prev):
                            bracket=np.array([z_prev,z1])
                            bracketF=np.array([f_prev,f1])
                            bracketG=np.array([g_prev,g1])
                            break
                        elif abs(gtd1)<=-0.9*gtd1:
                            bracket=[z1]
                            bracketF=[f1]
                            bracketG=[g1]
                            done=True
                            break
                        elif gtd1>0:
                            bracket=np.array([z_prev,z1])
                            bracketF=np.array([f_prev,f1])
                            bracketG=np.array([g_prev,g1])
                            break
                    temp=z_prev
                    z_prev=z1
                    minStep=z1+0.01*(z1-temp)
                    maxStep=10*z1
                    z1=self.poly(temp,z1,f_prev,f1,gtd_prev,gtd1,minStep,maxStep)
                    f_prev=f1
                    g_prev=g1
                    f1=self.getFunction(self.var+r*z1)
                    g1=self.getGradient(self.var+r*z1)
                    #print "Forward search",f1
                    gtd_prev=gtd1
                    gtd1=self.var_inner(g1,r)
                    lsIter=lsIter+1
                if lsIter==lsIterMax:
                    bracket=np.array([0,z1])
                    bracketF=np.array([f0,f1])
                    bracketG=np.array([g0,g1])
                #print "Forward Track:",lsIter
                insufProgress=False
                while not done and lsIter<lsIterMax:
                    f_Lo=np.min(bracketF)
                    LoPos=np.argmin(bracketF)
                    HiPos=1-LoPos
                    p1=bracket[0]
                    fp1=bracketF[0]
                    fg1=self.var_inner(bracketG[0],r)
                    p2=bracket[1]
                    fp2=bracketF[1]
                    fg2=self.var_inner(bracketG[1],r)
                    z1=self.poly(p1,p2,fp1,fp2,fg1,fg2,min(p1,p2),max(p1,p2))
                    if min(max(bracket)-z1,z1-min(bracket))/(max(bracket)-min(bracket)) < 0.1:
                        if insufProgress or z1>=max(bracket) or z1 <= min(bracket):
                            if abs(z1-max(bracket)) < abs(z1-min(bracket)):
                                z1 = max(bracket)-0.1*(max(bracket)-min(bracket))
                            else:
                                z1 = min(bracket)+0.1*(max(bracket)-min(bracket))
                            insufProgress=False
                        else:
                            insufProgress = True
                    else:
                        insufProgress=False
                    f_new=self.getFunction(self.var+r*z1)
                    g_new=self.getGradient(self.var+r*z1)
                    #print "Backtrack:", f_new
                    gtd_new=self.var_inner(g_new,r)
                    lsIter=lsIter+1
                    #print bracket
                    #print bracketF
                    armijo = f_new < f0 + 0.0001*z1*gtd0
                    if ~armijo or f_new >= f_Lo:
                        bracket[HiPos] = z1
                        bracketF[HiPos] = f_new
                        bracketG[HiPos] = g_new
                        Tpos = HiPos
                    else:
                        if abs(gtd_new) <= - 0.9*gtd0:
                            done=True
                        elif gtd_new*(bracket[HiPos]-bracket[LoPos]) >= 0:
                            bracket[HiPos] = bracket[LoPos]
                            bracketF[HiPos] = bracketF[LoPos]
                            bracketG[HiPos] = bracketG[LoPos]
                        bracket[LoPos] = z1
                        bracketF[LoPos] = f_new
                        bracketG[LoPos] = g_new
                        Tpos=LoPos
                f_Lo=np.min(bracketF)
                LoPos=np.argmin(bracketF)
                z1=bracket[LoPos]
                if np.isnan(f_Lo):
                    #print "Failed line search"
                    self.var=self.var-g0*0.00001
                    self.update_var()
                    self.last_z1=0.00001
                    s=g0*0.00001
                    self.S.append(s)
                else:
                    self.var=self.var+z1*r
                    self.update_var()
                    d=z1*r
                    s=d
                    self.S.append(s)
                    self.last_z1=max(z1,0.0001)
            else:
                self.var=self.var+r*0.01
                self.S.append(0.01*r)
                self.update_var()
            grad=self.getGradient(self.var+r*z1)
            self.S.remove(self.S[0])
            self.Y.remove(self.Y[0])
            self.YS.remove(self.YS[0])
            y=grad-self.old_grad
            self.Y.append(y)
            self.YS.append(self.var_inner(y,s))
            self.old_grad=grad
            self.NumIter=self.NumIter+1
            if z1<0.000001:
                z1=self.learningRate
            self.last_z1=z1
            return f_Lo,z1










