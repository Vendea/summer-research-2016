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
from SPSA import SPSA
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nbits=11
n_layer=3
n_nodes=8

# Specifying the inputs into the graph
y = tf.placeholder('float', [None, nbits+1])
x = tf.placeholder('float', [None, nbits*2])

# Defining and initializing the trainable Variables
weights = [tf.Variable(tf.random_normal([nbits*2, n_nodes]))]
biases = []
for i in range(1, n_layer):
    biases.append(tf.Variable(tf.random_normal([n_nodes])))
    weights.append(tf.Variable(tf.random_normal([n_nodes, n_nodes])))

weights.append(tf.Variable(tf.random_normal([n_nodes, nbits+1])))
biases.append(tf.Variable(tf.random_normal([n_nodes])))
biases.append(tf.Variable(tf.random_normal([nbits+1])))
base=2
add_layer=[]
inc_layer=[]
for i in range(n_nodes/2):
    add_layer.append(Adder(base))
    inc_layer.append((Incrementer(base)))

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
            else:
                for i in range(n_nodes/2):
                    a,b =  inc_layer[i].ex(tf.slice(aes, [0,i], [0,1]), tf.slice(bes, [0,i], [0,1]))
                    temp.append(a)
                    temp.append(b)
                layer.append(tf.concat(1,temp))
    return layer[-1]

# Defining the output
pred = multilayer_perceptron(x, weights, biases)


cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, pred))))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

saver = tf.train.Saver()

# Graph initialization
init = tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

data_x=[]
data_y=[]
g = prime.generate_data(nbits)
data_x, data_y = g[0]
test_x, test_y = g[1]
feed={x:data_x,y:data_y}

mini=SPSA(cost,feed,sess)
if rank==0:
    start=time.time()
    for n in range(1000):
        orig=mini.var
        g0=mini.getGrad(cost,n)
        g1=comm.recv(source=1,tag=11)
        g=[]
        for m1,m2 in zip(g0,g1):
            g.append((m1+m2)/2)
        f,update0=mini.minimize(g,orig,n)
        a0=sess.run(accuracy,feed)
        f0=sess.run(cost,feed)
        update1=comm.recv(source=2,tag=11)
        mini.set_var(update1)
        a1=sess.run(accuracy,feed)
        f1=sess.run(cost,feed)
        print time.time()-start,f1,f0,a1,a0
        if a1>a0:
            update=comm.bcast(update1,root=0)
        else:
            update=comm.bcast(update0,root=0)
        mini.set_var(update)
        if n%10==0:
            print sess.run(accuracy,{x:test_x,y:test_y})

elif rank==1:
    for n in range(1000):
        g=mini.getGrad(cost,n)
        comm.send(g,dest=0,tag=11)
        update=comm.bcast(None,root=0)
        mini.set_var(update)
elif rank==2:
    for n in range(1000):
        orig=mini.var
        g0=mini.getGrad(cost,n)
        g1=comm.recv(source=3,tag=11)
        g=[]
        for m1,m2 in zip(g0,g1):
            g.append((m1+m2)/2)
        f,update=mini.minimize(g,orig,n)
        comm.send(update,dest=0,tag=11)
        update=comm.bcast(None,root=0)
        mini.set_var(update)
elif rank==3:
    for n in range(1000):
        g=mini.getGrad(cost,n)
        comm.send(g,dest=2,tag=11)
        update=comm.bcast(None,root=0)
        mini.set_var(update)
        
'''for ep in range(1000):
    o1,n1=mini.minimize(cost,ep)
    f1=sess.run(cost,feed)
    mini.set_var(o1)
    o2,n2=mini.minimize(cost,ep)
    f2=sess.run(cost,feed)
    mini.set_var(o2)
    o3,n3=mini.minimize(cost,ep)
    f3=sess.run(cost,feed)
    if f1<=f2 and f1<=f3:
        mini.set_var(n1)
    elif f2<=f3 and f2<=f1:
        mini.set_var(n2)
    else:
        mini.set_var(n3)
    print sess.run(cost,feed)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: test_x, y: test_y},session=sess)

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict=feed))'''
