'''
Created on Jun 17, 2016
@author: Katherine Beine, Trae Hurley, Eyob Tsegaye, Mingsheng Wu

This is a distributed implementation of a neuralnets framework using tensorflow used to factorize large integers.
'''
import prime
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from Adder import Adder
from Incrementer import Incrementer
from mpi4py import MPI
import tensorflow as tf

nbits=11
n_layer=nbits
n_out=nbits+1
n_nodes=2*n_out

# Specifying the inputs into the graph
x = tf.placeholder('float', [None, nbits*2])
y = tf.placeholder('float', [None, nbits+1])

# Defining and initializing the trainable Variables
weights = [tf.Variable(tf.random_normal([nbits*2, n_nodes]))]
biases = []
for i in range(1, n_layer):
    biases.append(tf.Variable(tf.random_normal([n_nodes])))
    weights.append(tf.Variable(tf.random_normal([n_nodes, n_nodes])))

weights.append(tf.Variable(tf.random_normal([n_nodes, n_out])))
biases.append(tf.Variable(tf.random_normal([n_nodes])))
biases.append(tf.Variable(tf.random_normal([n_out])))
base=2
add_layer=[]
inc_layer=[]
for i in range(n_out):
    add_layer.append(Adder(base))
#    inc_layer.append((Incrementer(base)))

# Constructing the hidden and output layers
def multilayer_perceptron(x, weights, biases):
    layer = [x]
    for i in range(1,n_layer+2):
        layer.append(tf.nn.sigmoid(tf.add(tf.matmul(layer[i-1], weights[i-1]), biases[i-1])))
        if i == n_layer+1:
            continue
        else:
            aes, bes = tf.split(1, 2, layer[len(layer)-1])
            temp = []
            if i%2 == 1:
                for i in range(n_nodes/2):
                    a,b =  add_layer[i].ex(tf.slice(aes, [0,i], [0,1]), tf.slice(bes, [0,i], [0,1]))
                    temp.append(a)
                    temp.append(b)
                layer.append(tf.concat(1, temp))
    return layer[-1]

# Defining the output
pred = multilayer_perceptron(x, weights, biases)


cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, pred))))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

saver = tf.train.Saver()

# Graph initialization
init = tf.initialize_all_variables()
sess=tf.InteractiveSession()
sess.run(init)

data_x=[]
data_y=[]
a,b = prime.generate_data(nbits)
data_x.append(a[0][0])
print len(data_x)
data_y.append(a[1][0])
print len(data_y)
feed={x:data_x,y:data_y}
sess.run(pred,feed)
