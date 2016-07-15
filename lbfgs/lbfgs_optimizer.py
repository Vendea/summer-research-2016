__author__ = 'billywu'

import tensorflow as tf
import numpy as np
import math
import time
import flatten
import gc

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
<<<<<<< HEAD
            v.append(sess.run(t))
            self.assign_placeholders.append(tf.placeholder(shape=v[-1].shape,dtype="float32"))
            assign_op.append(t.assign(self.assign_placeholders[-1]))
        self.assign=tf.group(*assign_op)
=======
            v.append(t.eval(session=self.sess))
>>>>>>> origin/master
        self.var=np.array(v)
        np.save('var',self.var)
        comm.scatter(['Init' for i in range(size)],root=rank)
        self.gradient=tf.gradients(cost,tf.trainable_variables())
        self.learningRate=learning_rate
        self.old_grad=None

    def update(self,data_x,data_y,x,y,keep_prob):
        feed=[]
        s=len(data_x)/self.size
        for i in range(self.size):
            feed.append(("U",(data_x[i*s:(i+1)*s],data_y[i*s:(i+1)*s])))
        data=self.comm.scatter(feed,root=0)
        data_x,data_y=data[1]
        self.feed={x:data_x,y:data_y,keep_prob:1.0}



    def update_var(self,var=None):
        l=[]
        if var==None:
            var=self.var
        feed={}
        for t,v in zip(self.assign_placeholders,var):
            feed[t]=v
        self.sess.run(self.assign,feed)
        np.save('var',var)
        self.comm.scatter([("W",0) for i in range(self.size)])

    def var_self_inner(self,var_v1,useFlatten=False):
        start=time.time()
        self.innerEval=self.innerEval+1
        if not useFlatten:
            v=[]
            for m in var_v1:
                v=v+[x for x in np.nditer(m, op_flags=['readwrite'])]
            v=np.array(v)

            s=np.inner(v,v)
            return s
        else:
            v=np.array(flatten.flatten(var_v1)[0])
            data=[]
            size=len(v)/self.size
            for i in range(self.size-1):
                data.append(("IS",v[i*size:(i+1)*size]))
            data.append(("IS",v[(self.size-1)*size:len(v)]))
            self.comm.scatter(data,root=self.rank)
            ss=np.inner(data[self.rank][1],data[self.rank][1])
            ss=self.comm.gather(ss,root=self.rank)
            #print "Self Inner Time",time.time()-start
            return np.sum(ss)

    # Numpy unstructured inner product
    def var_inner(self,var_v1,var_v2, useFlatten=False):
        start=time.time()
        self.innerEval=self.innerEval+1
        if not useFlatten:
            s=time.time()
            v1=[]
            v2=[]
            for m1,m2 in zip(var_v1,var_v2):
                v1=v1+[x for x in np.nditer(m1, op_flags=['readwrite'])]
                v2=v2+[x for x in np.nditer(m2, op_flags=['readwrite'])]
            e=time.time()
            ret=np.inner(v1,v2)
            #print "Inner product:", e-s
            #print "Inner Time",time.time()-start
            return ret
        else:
            v1=flatten.flatten(var_v1)[0]
            v2=flatten.flatten(var_v2)[0]
            data=[]
            size=len(v1)/self.size
            for i in range(self.size-1):
                data.append(("I",(v1[i*size:(i+1)*size],v2[i*size:(i+1)*size])))
            data.append(("I",(v1[(self.size-1)*size:len(v1)],v2[(self.size-1)*size:len(v1)+1])))
            #print "Separate Time",time.time()-start
            self.comm.scatter(data,root=self.rank)
            ss=np.inner(data[self.rank][1][0],data[self.rank][1][1])
            ss=self.comm.gather(ss,root=self.rank)
            #print "Self Inner Time",time.time()-start
            return np.sum(ss)


    def update_hessian(self,g,Hdiag):
        self.HessianEval=self.HessianEval+1
        s=time.time()
        d=-g
        d,sh=flatten.flatten(d)
        d=np.array(d)
        al=np.zeros(self.memorySize)
        be=np.zeros(self.memorySize)
        for j in range(self.memorySize):
            i=self.memorySize-j-1
            al[i]=(np.inner(self.S[i],d))/self.YS[i]
            d=d-al[i]*self.Y[i]
        d=Hdiag*d
        for i in range(self.memorySize):
            be[i]=(np.inner(self.Y[i],d))/self.YS[i]
            d=d+self.S[i]*(al[i]-be[i])
        #print "Hessian Update:", time.time()-s
        return np.array(flatten.unflatten((d,sh)))



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
        gradients=self.comm.gather(data,root=self.rank)
        ret=gradients[0]
        for i in range(1,len(gradients)):
            ret=np.add(ret,gradients[i])
        e=time.time()
        #print "Gradient Time:",e-s
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
            s=np.array(flatten.flatten(-0.00001*grad)[0])
            y=np.array(flatten.flatten(grad-self.old_grad)[0])
            self.S.append(s)
            self.Y.append(y)
            self.YS.append(np.inner(self.S[-1],self.Y[-1]))
            self.old_grad=grad
            self.memorySize=self.memorySize+1
            self.NumIter=self.NumIter+1
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
                    if f1>f0+0.001*z1*gtd0 or (lsIter>1 and f1>f_prev):
                        bracket=np.array([z_prev,z1])
                        bracketF=np.array([f_prev,f1])
                        bracketG=np.array([g_prev,g1])
                        break
                    elif abs(gtd1)<=-0.9*gtd1:
                        bracket=z1
                        bracketF=f1
                        bracketG=g1
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
<<<<<<< HEAD
                        Tpos=LoPos
=======
                        Tpos = LoPos
>>>>>>> origin/master
                f_Lo=np.min(bracketF)
                LoPos=np.argmin(bracketF)
                z1=bracket[LoPos]
                self.var=self.var+z1*r
                self.update_var()
                d=z1*r
                s=np.array(flatten.flatten(d)[0])
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
            y=np.array(flatten.flatten(grad-self.old_grad)[0])
            self.Y.append(y)
            self.YS.append(np.inner(y,s))
            self.old_grad=grad
            self.NumIter=self.NumIter+1
            return f_Lo










