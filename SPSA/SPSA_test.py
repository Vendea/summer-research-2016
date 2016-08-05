# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from mpi4py import MPI
import time
import numpy as np
from SPSA import SPSA
import numpy
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_groups = 2

group =   [x for x in range(size) if x% num_groups == rank%num_groups]
parents = list(set([x% num_groups for x in range(size)]))
# Parameters
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

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess=tf.Session()
sess.run(init)


data_x, data_y = mnist.train.next_batch(10000)
feed={x:data_x,y:data_y}
mini=SPSA(cost,feed,sess)
if rank==0:
    start=time.time()
    for n in range(100000):
        orig=mini.var
        g=[[mini.getGrad(cost,n)]]
        group.remove(rank)
        for x in group:
            g.append([comm.recv(source=x,tag=11)]) 
        g = np.average(g,axis=0)
        c,update0=mini.minimize(g[0],orig,n)
        updates = [(update0,c)]
        for x in parents:
            updates.append(comm.recv(source=x,tag=11))
        c,updates = updates
        update=comm.bcast(updates[c.index(min(c))],root=0)
        mini.set_var(update)
elif rank in parents:
    for n in range(100000):
        orig=mini.var
        g=[[mini.getGrad(cost,n)]]
        group.remove(rank)
        for x in group:
            g.append(comm.recv(source=x,tag=11))
        g = np.average(g,axis=0)
        f,update=mini.minimize(g[0],orig,n)
        comm.send((update,f),dest=0,tag=11)
        update=comm.bcast(None,root=0)
        mini.set_var(update)
else :
    for n in range(100000):
        g=mini.getGrad(cost,n)
        comm.send(g,dest=rank%num_groups,tag=11)
        update=comm.bcast(None,root=0)
        mini.set_var(update)





