
__author__ = 'billywu'

from mpi4py import MPI
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import tensorflow as tf
import time
from lbfgs_optimizer import lbfgs_optimizer
from Opserver import Opserver
import numpy as np

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
# Launch the graph
config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 1},
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
sess=tf.Session(config=config)
sess.run(init)
train_size=50000
tx,ty=batch_xs, batch_ys = mnist.train.next_batch(train_size)
bsize=4
total_time=0
if rank==0:
    trainer=lbfgs_optimizer(0.0001, cost,[],sess,1,comm,size,rank)
    for b in range(100):
        data_x=tx[bsize*b:bsize*(b+1)]
        data_y=ty[bsize*b:bsize*(b+1)]
        trainer.update(data_x,data_y,x,y,keep_prob)
        trainer.memorySize=0
        trainer.NumIter=0
        trainer.S=[]
        trainer.Y=[]
        trainer.YS=[]
        start=time.time()
        i=0
        while i<100:
            c,s= trainer.minimize()
            i=i+1
            print c,s
            if i%10==0:
                print "All Performance"
                time_cost=time.time()-start
                total_time=time_cost+total_time
                train=sess.run(accuracy,{x:data_x,y:data_y,keep_prob:1.0})
                test= sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
                train_cost=c
                f=trainer.functionEval
                g=trainer.gradientEval
                inner=trainer.innerEval
                print total_time, time_cost, inner, f, g, train, test,train_cost,s
                start=time.time()
            if c<10 and not s==0:
                print "Zero cost"
                trainer.last_z1=trainer.learningRate
                break

else:
    opServer=Opserver(0.0001, cost,[],sess,comm,size,rank,0,x,y,keep_prob)
    opServer.run()




