__author__ = 'billywu'

from mpi4py import MPI
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import time
import numpy as np
from SPSA import SPSA

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = multilayer_perceptron(x, weights, biases)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess=tf.Session()
sess.run(init)


data_x, data_y = mnist.train.next_batch(20000)
feed={x:data_x,y:data_y}
mini=SPSA(cost,feed,sess)
simulation_steps=10
num_offsrping=2
ep=0

while True:
    for i in range(simulation_steps):
        o,n=mini.minimize(cost,ep)
        ep=ep+1
    f=accuracy.eval(feed,session=sess)
    fs = np.array(comm.gather(f,root=0))
    if rank==0:
        msg=[]
        best=np.argmax(fs)
        for i in range(size):
            msg.append(best)
    else:
        msg=[]
    best=comm.scatter(msg,root=0)
    if best==rank:
        msg=[n for i in range(size)]
    else:
        msg=[]
    msg=comm.scatter(msg,root=best)
    mini.set_var(msg)
    if rank==3:
        print sess.run(cost,feed)
        print "Accuracy:", accuracy.eval(feed,session=sess)






