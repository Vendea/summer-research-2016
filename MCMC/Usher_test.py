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
from Usher import MCMC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import math

master = MPI.COMM_WORLD.Get_rank() == 0

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

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess=tf.Session()
sess.run(init)


data_x, data_y = mnist.train.next_batch(10000)
feed={x:data_x,y:data_y}
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
mini=MCMC(accuracy, {x: mnist.test.images, y:mnist.test.labels}, sess, stdev=0.035, maximize=True)
#mini=MCMC(accuracy,{x: mnist.test.images, y:mnist.test.labels},sess,0,MPI.COMM_WORLD)
costs = []
costs.append(mini.prev_cost)
timestamps = [0]
start = time.time()
runs = 1
num_epochs_unchanged = 0
while time.time()-start < 72000: #for ep in range(1000):
    if num_epochs_unchanged == 500:
        mini.optimize(mini.stdev / 2.0)
        num_epochs_unchanged = 0
    else:
        mini.optimize()
    timestamps.append(time.time() - start)
    costs.append(mini.prev_cost)
    if master:
        print time.time()-start, mini.prev_cost
    #if time.time()-start > 300:
    #    break
    #print sess.run(cost, feed)
    #print "Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels},session=sess)
    runs += 1
    if costs[-2] == costs[-1]:
        num_epochs_unchanged += 1
    else:
        num_epochs_unchanged = 0

#print timestamps
#print costs
plt.plot(timestamps, costs, label='Usher')
plt.legend(bbox_to_anchor=(.9,.5), bbox_transform=plt.gcf().transFigure)
plt.grid(True)
plt.savefig('../pics/usher/usher_stdev'+str(mini.stdev)[2:]+'_'+str(MPI.COMM_WORLD.Get_rank())+'.png')
