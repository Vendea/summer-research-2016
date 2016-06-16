__author__ = 'billywu'

import tensorflow as tf
import numpy as np
import math

start=True
sess = tf.InteractiveSession()


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200.0*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400.0*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200.0*(x[-1]-x[-2]**2)
    return der




def minimize(x):
    rho=0.01
    sig=0.5
    INT=0.1
    ext=3.0
    nmax=20
    ratio=100
    i=0
    red=1
    ls_failed=False
    fx=[]
    f1,df1=rosen(x),rosen_der(x)
    evaluate=1
    s=-df1
    d1=-np.inner(s,s)
    z1=red/(1-d1)
    counter=0
    while np.linalg.norm(df1)>1:
        counter=counter+1
        x0=x
        f0=f1
        df0=df1
        x=x+z1*s
        f2,df2=rosen(x),rosen_der(x)
        evaluate=evaluate+1
        d2=np.inner(df2,s)
        f3=f1
        d3=d1
        z3=-z1
        M=nmax
        success=0
        limit=-1
        while (True):
            while ((f2>f1+z1*rho*d1 or (d2 > -sig*d1) and (M > 0))):
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
                x=x+z2*s
                f2,df2=rosen(x),rosen_der(x)
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
            x=x+z2*s
            f2,df2=rosen(x),rosen_der(x)
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
            x=x0
            f1=f0
            df1=df0
            if ls_failed==1:
                break
            tmp=df1
            df1=df2
            df2=tmp
            s=-df1
            d1=-np.inner(s,s)
            z1=1/(1-d1)
            ls_failed =1
    print counter
    print evaluate
    return x

x=np.array([1,2,3])
print rosen_der(x)