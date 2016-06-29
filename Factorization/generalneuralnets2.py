'''
Created on Jun 17, 2016
@author: Katherine Beine, Trae Hurley, Eyob Tsegaye, Mingsheng Wu

This is a distributed implementation of a neuralnets framework using tensorflow used to factorize large integers.
'''
import prime
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
'''===============================================================================================
    Preprocessing and Configurations
   ===============================================================================================
'''
# Configuring IPC:
#   1. initializing MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#   2. specifying the number of cores allocated
num_cores = 4
#   3. specifying the core identification numbers
l_cores = [0,1,2,3]
#   4. determining the master/slave model
counter = 0
l_cores = sorted(l_cores)
master = l_cores[0]
slaves = l_cores[1:num_cores]
core_dict = {}
for c in slaves:
    core_dict[c] = counter
    counter += 1

# Configuring NN Structure:
#   1. predefined number of layers
n_layer = 3
#   2. predefined number of nodes per hidden layer
n_nodes = 128
#   3. cost function select
#           (used to study how the different cost function
#            can lead to different convergence properties):
#       0: RMSE
#       1: Cross Entropy
#       2: Absolute factor differences
#       3: bitwise differences
cost_func = 0
#   4. kernel select:
#       0: sigmoid
#       1: relu
#       2: tanh
#       3: softmax
kernel=0
activation = {0:tf.nn.sigmoid, 1:tf.nn.relu, 2:tf.nn.tanh, 3:tf.nn.softmax}

# Configuring Learning Properties:
#   1. learning rate
learning_rate = .0001
#   2. learning epochs
training_epochs = 10000
#   3. mini-batch sizes
batch_size = 100
#   4. evaluating frequency
display_step=1


# Configuring Problem Specific Data Setup
#   1. number of bits of the prime factors
nbits = 16
#   2. training size
training_sizes = 5000
#   3. testing size
testing_sizes = 4000
#   4. obtaining the data
data_x, data_y = prime.generate_data(nbits)
#   5. splitting the training and testing data
train_x = data_x[0:training_sizes]
test_x = data_x[-(testing_sizes + 1):-1]
train_y = data_y[0:training_sizes]
test_y = data_y[-(testing_sizes + 1):-1]

logger.info("Data preprocessing done in core"+str(rank))

'''===============================================================================================
    Neural Network Model Tensorflow Replicas Construction
   ===============================================================================================
'''

# Specifying the inputs into the graph
x = tf.placeholder('float', [None, nbits*2])
y = tf.placeholder('float', [None, nbits])

# Defining and initializing the trainable Variables
weights = [tf.Variable(tf.random_normal([nbits*2, n_nodes]))]
biases = []
for i in range(1, n_layer):
    biases.append(tf.Variable(tf.random_normal([n_nodes])))
    weights.append(tf.Variable(tf.random_normal([n_nodes, n_nodes])))

weights.append(tf.Variable(tf.random_normal([n_nodes, nbits])))
biases.append(tf.Variable(tf.random_normal([n_nodes])))
biases.append(tf.Variable(tf.random_normal([nbits])))

# Constructing the hidden and output layers
def multilayer_perceptron(x, weights, biases):
    layer = [x]
    for i in range(1,n_layer+2):
        layer.append(activation[kernel](tf.add(tf.matmul(layer[i-1], weights[i-1]), biases[i-1])))
    return layer[-1]

# Defining the output
pred = multilayer_perceptron(x, weights, biases)

# Defining the cost function based on the function select
if cost_func == 0:
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, pred))))
elif cost_func == 1:
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
elif cost_func == 2:
    c = np.zeros([nbits,1]).astype('float32')
    for i in range(nbits):
        c[i, 0] = 2**i
    conv = tf.Variable(c)
    tpred = tf.nn.sigmoid(tf.mul(1000.0, pred))
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tf.matmul(tpred, conv), tf.matmul(y, conv)))))
else:
    c = np.ones([nbits,1]).astype('float32')
    conv = tf.Variable(c)
    tpred = tf.nn.sigmoid(tf.mul(1000.0, pred))
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tf.matmul(tpred, conv), tf.matmul(y, conv)))))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()

# Graph initialization
init = tf.initialize_all_variables()
sess=tf.InteractiveSession()
sess.run(init)
logger.info("Network initialization done in core"+str(rank))

'''===============================================================================================
    Distributed Training and Model evaluation
   ===============================================================================================
'''

# Master core code
if rank == master:
    timer = 0
    print "master started..."
    # Constructing the initial message
    data_w = []
    data_b = []
    for w,b in zip(weights, biases):
        data_w.append(w.eval())
        data_b.append(b.eval())
    data = (data_w, data_b)
    print "master core started..."
    for epoch in range(training_epochs):
        start_time = time.time()
        avg_cost = 0
        total_batch = int(training_sizes/batch_size)
        # distributing tasks to the following slave cores
        comm.bcast(data, root=0)
        # collecting the weights and biases obtained from each slave cores
        weights_n = []
        biases_n = []
        data_n = []
        empty_weight = [np.zeros([nbits*2, n_nodes])]
        empty_bias = []
        for _ in range(1, n_layer):
            empty_weight.append(np.zeros([n_nodes, n_nodes]))
            empty_bias.append(np.zeros(n_nodes))
        empty_weight.append(np.zeros([n_nodes, nbits]))
        empty_bias.append(np.zeros(n_nodes))
        empty_bias.append(np.zeros(nbits))
        data = tuple(comm.reduce(np.array([empty_weight, empty_bias]), op=MPI.SUM, root=0) / len(slaves))

        avg_weight,avg_bias = data
        # evaluating the training progress every display_step epochs
        if (epoch % display_step) == 0:
            for w,t in zip(avg_weight, weights):
                sess.run(t.assign(w))
            for b,t in zip(avg_bias, biases):
                sess.run(t.assign(b))
            # evaluating on the cost function
            test_cost = sess.run(cost,{x:test_x, y:test_y})
            #print "epoch",epoch,"test_cost:", test_cost
            train_cost = sess.run(cost, {x:train_x, y:train_y})
            print "epoch",epoch,"train_cost:", train_cost
            train_pred = sess.run(pred, {x:train_x, y: train_y})
            # evaluating on the actual accuracy
            correct = 0
            for tp, ty in zip(train_pred, train_y):
                valid = True
                for tpe, tye in zip(tp, ty):
                    if tpe > 0 and tye == -1:
                        valid = False
                        break
                    elif tpe <= 0 and tye ==1:
                        valid = False
                        break
                if valid:
                    correct = correct +1
            #print "epoch", epoch, "training accuracy:", (correct + 0.0)/training_sizes
            test_pred = sess.run(pred, {x:test_x, y: test_y})
            correct = 0
            for tp, ty in zip(test_pred, test_y):
                valid = True
                for tpe, tye in zip(tp, ty):
                    if tpe > 0 and tye == -1:
                        valid = False
                        break
                    elif tpe <= 0 and tye ==1:
                        valid = False
                        break
                if valid:
                    correct = correct +1
            #print "epoch", epoch, "testing accuracy:", (correct + 0.0)/testing_sizes
        end_time = time.time()
        timer += end_time-start_time
        #if (epoch % display_step) == 0:
            #logger.info("time taken for epoch"+str(epoch)+":"+str(end_time-start_time))
    print "avg time per epoch:", timer/training_epochs
    print "master is done"

# Slave core code
else:
    local_size = training_sizes / (num_cores-1)
    p = core_dict[rank]
    # Initializing local data shard
    local_train_x = train_x[p*local_size:(p+1)*local_size]
    local_train_y = train_y[p*local_size:(p+1)*local_size]

    for epoch in range(training_epochs):
        # waiting on weight initialization
        data = None
        data = comm.bcast(data, root=0)
        data_w,data_b = data
        # apply the variables
        for w,t in zip(data_w, weights):
            sess.run(t.assign(w))
        for b,t in zip(data_b, biases):
            sess.run(t.assign(b))
        # batching training
        for i in range(local_size/batch_size):
            batch_x=local_train_x[i*batch_size:(i+1)*batch_size]
            batch_y=local_train_y[i*batch_size:(i+1)*batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        data_w = []
        data_b = []
        for w,b in zip(weights, biases):
            data_w.append(w.eval())
            data_b.append(b.eval())
        # return slave results
        comm.reduce(np.array([data_w, data_b]), op=MPI.SUM, root=0)
        #comm.send((data_w, data_b), dest=master, tag=11)

    print "core", rank, "(slave) is done"
