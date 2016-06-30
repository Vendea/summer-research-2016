'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import time
from DistributedSPSA import SPSA
from mpi4py import MPI
import sys
from sys import path
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)
import Logger as l

args = len(sys.argv)
comm_children = None
comm_parent   = None
train_cost    = None
test_cost     = None
correct_prediction,accuracy = None,None
train_cost,train_accuracy = [],[]
test_cost,test_accuracy = [],[]
# Network Parameters
learning_rate = 0.001
training_epochs = 600
batch_size   = 100
n_layer      = 2
n_nodes      = 256
n_input      = 784 # MNIST data input (img shape: 28*28)
n_classes    = 10 # MNIST total classes (0-9 digits)
n_workers    = 2
gen_size     = 1
train_size   = 4000
start_worker = "worker"
#random stuff
weights = []
biases = []
pred = None
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

if args == 1:
    comm_global = MPI.COMM_WORLD
    rank = comm_global.Get_rank()
    size = comm_global.Get_size()
    comm_children = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[sys.argv[0], start_worker],
        maxprocs=n_workers*size)
    start=n_workers*rank
    end = start+n_workers -1
    if rank == 0:
        f = l.DataLogger("SPSA",n_layer,n_nodes)
else:
    comm_global = MPI.COMM_WORLD
    comm_parent = MPI.Comm.Get_parent()
    rank = comm_parent.Get_rank()
    size = comm_parent.Get_size()
# Constructing the hidden and output layers
def multilayer_perceptron():
    prevlayer = x
    for w,b in zip(weights,biases):
        prevlayer = tf.nn.relu(tf.add(tf.matmul(prevlayer, w), b))
    return prevlayer
def build_network_master():
    global pred,accuracy
  
    # Defining and initializing the trainable Variables
    weights.append(tf.Variable(tf.random_normal([n_input, n_nodes])))
    for i in range(1, n_layer):
        biases.append(tf.Variable(tf.random_normal([n_nodes])))
        weights.append(tf.Variable(tf.random_normal([n_nodes, n_nodes])))
    weights.append(tf.Variable(tf.random_normal([n_nodes, n_classes])))
    biases.append(tf.Variable(tf.random_normal([n_nodes])))
    biases.append(tf.Variable(tf.random_normal([n_classes])))
    # Defining the output
    data_w = []
    data_b = []
    init = tf.initialize_all_variables()
    # Launch the graph
    sess.run(init)
    for w,b in zip(weights, biases):
        data_w.append(w.eval())
        data_b.append(b.eval())

    data = zip(data_w, data_b)
    for i in range(1,size):
        comm_global.send(data, dest=i,tag=11)
    for i in range(start,end):
        comm_children.send(data, dest=i,tag=11)
    pred = multilayer_perceptron()

    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   
def build_network_slaves():
    global pred,accuracy
    data = None
    data = comm_global.recv(source=0,tag =11)
    for i in range(start,end):
        comm_children.send(data, dest=i,tag=11)

    # Defining and initializing the trainable Variables
    for w,b in data:
        biases.append(tf.Variable(b))
        weights.append(tf.Variable(w))
    # Defining the output
    pred = multilayer_perceptron()
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    init = tf.initialize_all_variables()
    # Launch the graph
    sess.run(init)

def build_network_children():
    global pred
    data = None
    data = comm_parent.recv(source=rank/n_workers,tag =11)
    # Defining and initializing the trainable Variables
    for w,b in data:
        biases.append(tf.Variable(b))
        weights.append(tf.Variable(w))
    # Defining the output
    pred = multilayer_perceptron()
    init = tf.initialize_all_variables()
    # Launch the graph
    sess.run(init)

def update_weights_slaves(c,update):
    comm_global.send([c,update,], dest=0, tag=11)
    data = comm_global.recv(source=0, tag=11)
    mini.set_var(data)
    update_weights_parent(data)

def update_weights_master(c,update):
    data,ls = [],[]
    del train_cost[:]
    for i in range(1,size):
            ls.append(comm_global.recv(source=i, tag=11))
    update_cost,update_values = zip(*ls)
    update_cost = list(update_cost)
    update_values = list(update_values)
    update_cost.append(c)
    update_values.append(update)
    while len(data) < gen_size:
        i = update_cost.index(max(update_cost))
        train_cost.append(update_cost.pop(i))
        data.append(update_values[i])
        update_values.pop(i)
    for i in range(size):
        if i != 0:
            comm_global.send(data[i%gen_size],dest=i, tag=11)
        else:
            mini.set_var(data[i%gen_size])
            update_weights_parent(data[i%gen_size])

def update_weights_children(g):
    comm_parent.send([g], dest=rank/n_workers, tag=11)
    mini.set_var(comm_parent.recv(source=rank/n_workers, tag=11))

def update_weights_parent(data):
    for i in range(start,end):
            comm_children.send(data,dest=i, tag=11)

with tf.Session() as sess:
    if rank == 0 and comm_parent == None:
        build_network_master()
    elif comm_parent == None:
        build_network_slaves()
    else:
        build_network_children()

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    # Initializing the variables
   

    data_x, data_y = mnist.train.next_batch(train_size)
    test_x, test_y = mnist.test.next_batch(train_size)
    feed={x:data_x,y:data_y}
    mini=SPSA(cost,feed,biases+weights,sess)

    for ep in range(training_epochs):
        timer = 0
        start_time = time.time()
        g = mini.nabla(cost,ep)
        if comm_parent != None:
            update_weights_children(g)
        else:
            dv = []
            for i in range(start,end):
                dv.append(comm_children.recv(source=i,tag=11))
            orginal,update = mini.minimize(dv,ep)
            c=sess.run(accuracy,feed)
            if rank != 0:
                update_weights_slaves(c,update)
            else:
                update_weights_master(c,update)
            
        end_time = time.time()
        if args == 1 and rank == 0:
            _,train_accuracy= sess.run([cost,accuracy],{x:data_x,y:data_y})
            _,test_accuracy = sess.run([cost,accuracy],{x:test_x,y:test_y})
            f.writeData(ep,
                        train_cost,
                        test_cost,
                        end_time-start_time,
                        train_accuracy,
                        test_accuracy)
            timer += end_time-start_time
            print("time taken for epoch"+str(ep)+":"+str(end_time-start_time))