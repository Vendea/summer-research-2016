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

def convert(x):
    return convert(x,2)
def convert(m,n):
    if(m<n):
        return [m]
    else:
        return convert(int(m/n),n) + [m%n]

for i in range(0,250):
    traindata += ((convert(a[i],2), convert(b[i],2)), convert(trainingSums[i],2))
    testdata += ((convert(a[i]+250,2), convert(b[i]+250,2)), convert(testingSums[i],2))
