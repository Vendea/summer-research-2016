import random as rand
import math as m
#import tensorflow as tf

# set up arrays of integers and sums for training and testing
a = [m.floor(rand.random()*250) for i in range(0, 250)]
b = [m.floor(rand.random()*250) for i in range(0, 250)]
trainingSums = [v+u for v in a for u in b]
testingSums = [500+v+u for v in a for u in b]

traindata = []
testdata = []

def dconvert(x):
    rval = convert(x,2)
    pad = [0 for i in range(10-len(rval))]
    return pad + rval

def convert(m,n):
    if(m<n):
        return [int (m)]
    else:
        return convert(int(m/n),n) + [int(m%n)]

for i in range(0,250):
    traindata += ((dconvert(a[i]), dconvert(b[i])),dconvert(trainingSums[i]))
    testdata += ((dconvert(a[i]+250), dconvert(b[i]+250)),dconvert(testingSums[i]))

print a[0], ",",b[0]
print traindata[0], ",", traindata[1]
