__author__ = 'billywu'

import tensorflow as tf
import numpy as np

class lbfgs_optimizer:
    def __init__(self,learning_rate, cost,feed,sess,m):
        self.Y=[]
        self.S=[]
        self.rho=[]
        self.cost=cost
        self.feed=feed
        self.sess=sess
        self.NumIter=0
        self.m=m
        self.memorySize=0
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

    def updateHessian(self,grad):
        q=grad
        alpha=np.zeros(self.memorySize)
        for i in range(self.memorySize-1,-1,-1):
            alpha[i]=self.rho[i]*self.var_inner(q,self.S[i])
            np.subtract(q,alpha[i]*self.Y[i],q)
        H0=self.var_inner(self.S[-1],self.Y[-1])/self.var_self_inner(self.Y[-1])
        r=np.multiply(H0,q)
        for i in range(self.memorySize-1):
            Beta=self.rho[i]*self.var_inner(self.Y[i],r)
            r=r+self.S[i]*alpha[i]-Beta
        return r

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

    def strong_wolfe_condition(self,alpha,r,var,c1,c2):
        self.update_var()
        cost_xk=self.sess.run(self.cost,self.feed)
        grad_xk=self.old_grad
        xk1=np.add(self.var,r*alpha)
        self.update_var(xk1)
        grad_xk1=np.array(self.sess.run(self.gradient,self.feed))
        cost_xk1=self.sess.run(self.cost,self.feed)
        return cost_xk1<cost_xk+c1*alpha*self.var_inner(r,grad_xk) and abs(self.var_inner(r,grad_xk1))>c2*abs(self.var_inner(r,grad_xk))


    def line_search(self, alpha,r):
        limit=0
        while not self.strong_wolfe_condition(alpha,r,self.var,0.0001,0.9) and limit<20:
            alpha=alpha/2
            limit=limit+1
        if limit==20:
            print "limit exceeded"
            return False
        else:
            xk1=np.add(self.var,r*alpha)
            self.update_var(xk1)
            self.var=xk1
            self.S.append(r*alpha)
            self.update_var()
            return True



    def minimize(self):
        if self.NumIter==0:
            grad=np.array(self.sess.run(self.gradient,self.feed))
            self.old_grad=grad
            grad=grad*(-self.learningRate)
            np.add(self.var,grad,self.var)
            self.S.append(grad)
            self.update_var()
            grad=np.array(self.sess.run(self.gradient,self.feed))
            self.Y.append(np.subtract(grad,self.old_grad))
            self.rho.append(1/self.var_inner(self.Y[-1],self.S[-1]))
            self.old_grad=grad
            self.memorySize=1
            self.NumIter=self.NumIter+1
            return self.var_self_inner(self.S[-1])
        else:
            r=-self.updateHessian(self.old_grad)
            if not self.line_search(self.learningRate,r):
                self.var=self.var-self.learningRate*self.old_grad
                self.S.append(-self.learningRate*self.old_grad)
                self.update_var()
            grad=np.array(self.sess.run(self.gradient,self.feed))
            self.Y.append(np.subtract(grad,self.old_grad))
            self.rho.append(1/self.var_inner(self.Y[-1],self.S[-1]))
            self.old_grad=grad
            if self.memorySize<=self.m:
                self.memorySize=self.memorySize+1
            else:
                self.Y.remove(self.Y[0])
                self.S.remove(self.S[0])
                self.rho.remove(self.rho[0])
            self.NumIter=self.NumIter+1
            return self.var_self_inner(self.S[-1])







