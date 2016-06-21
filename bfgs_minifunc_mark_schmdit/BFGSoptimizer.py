import math
import numpy as np
import tensorflow as tf
import flatten

class BFGSoptimizer:
    def __init__(self,cost,feed,var_t,sess):
        self.cost=cost                  # cost function to be minimized
        self.feed=feed                  # feed variables
        self.var=[]                     # list of dictionary containing the trainable variables
        w=[]                            # this part will convert the variables(tensors) into a list of real numbers(x)
        self.sess=sess
        for tl in var_t:
            for t in tl:
                self.var.append(tl[t])
                w.append(tl[t].eval(session=sess))
        self.x,self.shape=np.array(flatten.flatten(w))      # shape of the variables used to update the values of the
                                                            # tensors


    def getGradient(self,cost,x):                                # based on the current x, update the cost
        sess=self.sess
        self.assign_x(x)
        var_grad = tf.gradients(cost,self.var)
        vg=sess.run(var_grad, feed_dict=self.feed)
        c=sess.run(cost, feed_dict=self.feed)
        g,shape=flatten.flatten(vg)
        return c,np.array(g)

    def update_feed(self,feed):                                 # change the feed variable
        self.feed=feed

    def assign_x(self,x):                                       # assign the x value to the tensors from a list
        self.x=x
        wts=flatten.unflatten((x,self.shape))
        for t,w in zip(self.var,wts):
            self.sess.run(t.assign(w))

    def minimize(self, cost,epoch,alpha,verbose=True,cap=0):
        rho=0.01
        sig=0.5
        INT=0.1
        ext=3.0
        nmax=20
        ratio=100
        red=1
        ls_failed=False
        f1,df1=self.getGradient(cost,self.x)
        evaluate=1
        s=-df1
        d1=-np.inner(s,s)
        z1=red/(1-d1)
        counter=0
        lc=0
        ls_limit=5
        f2=f1
        while evaluate<epoch and np.linalg.norm(df1)>0.0000001 and f2>cap:
            counter=counter+1
            x0=self.x
            f0=f1
            df0=df1
            self.x=self.x+z1*s
            self.result=self.x
            f2,df2=self.getGradient(cost,self.x)
            evaluate=evaluate+1
            d2=np.inner(df2,s)
            f3=f1
            d3=d1
            z3=-z1
            M=nmax
            success=0
            limit=-1
            while (True):
                while ((f2>f1+z1*rho*d1 or (d2 > -sig*d1) and (M > 0)) and lc<ls_limit):
                    lc=lc+1
                    limit=z1
                    if f2>f1:
                        z2=z3-(0.5*d3*z3*z3)/(d3*z3+f2-f3)
                    else:
                        A = 6*(f2-f3)/z3+3*(d2+d3)
                        B = 3*(f3-f2)-z3*(d3+2*d2)
                        z2= (math.sqrt(B*B-A*d2*z3*z3)-B)/A
                    if np.isinf(z2) or np.isnan(z2):
                        z2=z3/2
                    z2=max(min(z2, INT*z3),(1-INT)*z3)
                    z1=z1+z2
                    self.x=self.x+z2*s
                    f2,df2=self.getGradient(cost,self.x)
                    evaluate=evaluate+1
                    M=M-1
                    d2=np.inner(df2,s)
                    z3=z3-z2
                if f2>f1+z1*rho*d1 or d2>-sig*d1:
                    break
                elif d2>sig*d1:
                    success=1
                    break
                elif M==0:
                    break
                A = 6*(f2-f3)/z3+3*(d2+d3)
                B = 3*(f3-f2)-z3*(d3+2*d2)
                valid=True
                try:
                    z2=-d2*z3*z3/(B+math.sqrt(B*B-A*d2*z3*z3))
                except:
                    valid=False
                if (not valid) or (not np.isreal(z2)) or (np.isnan(z2)) or np.isinf(z2):
                    if limit<-0.5:
                        z2=z1*(ext-1)
                    else:
                        z2=(limit-z1)/2
                elif (limit>-0.5) and (z2+z1>limit):
                    z2=z1*(limit-z1)/2
                elif (limit<-0.5) and (z2+z1>z1*ext):
                    z2=z1*(ext-1.0)
                elif z2<-z3*INT:
                    z2=-z3*INT
                elif (limit>-0.5) and (z2<(limit-z1)*(1.0-INT)):
                    z2=(limit-z1)*(1.0-INT)
                f3=f2
                d3=d2
                z3=-z2
                z1=z1+z2
                self.x=self.x+z2*s
                f2,df2=self.getGradient(cost,self.x)
                evaluate=evaluate+1
                M=M-1
                d2=np.inner(df2,s)
            if success==1:
                f1=f2
                s = (np.inner(df2,df2)-np.inner(df1,df2))/(np.inner(df1,df1))*s - df2
                tmp = df1; df1 = df2; df2 = tmp
                d2=np.inner(df1,s)
                if d2>0:
                    s=-df1
                    d2=np.inner(-s,s)
                z1=z1 * min(ratio, d1/(d2-0.00000000000000001))
                d1 = d2
                ls_failed = 0
            else:
                self.x=x0
                f1=f0
                df1=df0
                if ls_failed==1:
                    self.x=self.x
                tmp=df1
                df1=df2
                df2=tmp
                s=-df1
                d1=-np.inner(s,s)
                z1=1/(1-d1)
                ls_failed =1
        self.assign_x(self.x)





