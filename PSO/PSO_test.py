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
import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from PSO_basic import PSO

master = MPI.COMM_WORLD.Get_rank() == 0

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
b_lo = -1
b_up = 1

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
    'h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1], b_lo, b_up)),
    'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], b_lo, b_up)),
    'out': tf.Variable(tf.random_uniform([n_hidden_2, n_classes], b_lo, b_up))
}
biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden_1], b_lo, b_up)),
    'b2': tf.Variable(tf.random_uniform([n_hidden_2], b_lo, b_up)),
    'out': tf.Variable(tf.random_uniform([n_classes], b_lo, b_up))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess=tf.Session()
sess.run(init)


data_x, data_y = mnist.train.next_batch(10000)
feed={x:data_x,y:data_y}
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
mini=PSO(MPI.COMM_WORLD, cost, {x: mnist.test.images, y:mnist.test.labels}, sess, b_up, b_lo, omega=0.2, phi_p=0.4, phi_g=0.4)
costs = []
costs.append(mini.eval_g_best)
timestamps = [0]
start = time.time()
while time.time()-start < 3600: #for ep in range(1000):
    mini.optimize()
    timestamps.append(time.time() - start)
    costs.append(mini.eval_my_best)
    if master:
        print time.time()-start, mini.eval_g_best
    #if time.time()-start > 300:
    #    break
    #print sess.run(cost, feed)
    #print "Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels},session=sess)

#print timestamps
#print costs
plt.plot(timestamps, costs, label='PSO')
plt.legend(bbox_to_anchor=(.9,.5), bbox_transform=plt.gcf().transFigure)
plt.grid(True)
plt.savefig('../pics/pso/pso_basic_o'+str(mini.omega)+'_p'+str(mini.phi_p)+'_g'+str(mini.phi_g)+'_'+str(MPI.COMM_WORLD.Get_rank())+'.png')
