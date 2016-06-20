'''
Created on Jun 17, 2016
@author: Katherine Beine, Tray Hurley, Eyob Tsegaye, Mingsheng Wu

This is a distributed implementation of a neuralnets framework using tensorflow used to factorize large integers.
'''
import prime
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
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
#   3. determining the master/slave model
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
n_nodes = 256
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
#       3. softmax
kernel=0
activation = {0:tf.nn.sigmoid, 1:tf.nn.relu, 2:tf.nn.tanh, 3:tf.nn.softmax}


# Configuring Learning Properties:
#   1. learning rate
learning_rate = .0001
#   2. learning epochs
training_epochs = 10000
#   3. mini-batch sizes
batch_size = 1000
#   4. evaluating frequency
display_step=100


# Configuring Problem Specific Data Setup
#   1. number of bits of the prime factors
nbits = 16
#   2. training size
training_sizes = 50000
#   3. testing size
testing_sizes = 40000
#   4. obtaining the data
data_x, data_y = prime.generate_data(nbits)
#   5. splitting the training and testing data
train_x = data_x[0:training_sizes]
test_x = data_x[-(testing_sizes + 1):-1]
train_y = data_y[0:training_sizes]
test_y = data_y[-(testing_sizes + 1):-1]

print "Data preprocessing done in core", rank

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
print "Network initialization done in core", rank

'''===============================================================================================
    Distributed Training and Model evaluation
   ===============================================================================================
'''
# Master core code
if rank == master:
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
        avg_cost = 0
        total_batch = int(training_sizes/batch_size)
        # distributing tasks to the following slave cores
        for i in slaves:
            comm.send(data, dest=i, tag=11)
        # collecting the weights and biases obtained from each slave cores
        weights_n = []
        biases_n = []
        data_n = []
        for i in slaves:
            data = comm.recv(source=i, tag=11)
            weights_n.append(data[0])
            biases_n.append(data[1])
        # averaging the changes
        avg_weight = np.average(weights_n, axis=0)
        avg_bias = np.average(biases_n, axis=0)
        data=(avg_weight,avg_bias)
        print "Average computed for epoch", epoch
        # evaluating the training progress every display_step epochs
        if (epoch % display_step) == 0:
            for w,t in zip(avg_weight, weights):
                sess.run(t.assign(w))
            for b,t in zip(avg_bias, biases):
                sess.run(t.assign(b))
            # evaluating on the cost function
            test_cost = sess.run(cost,{x:test_x, y:test_y})
            print "epoch",epoch,"test_cost:", test_cost
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
            print "epoch", epoch, "training accuracy:", (correct + 0.0)/training_sizes
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
            print "epoch", epoch, "testing accuracy:", (correct + 0.0)/testing_sizes
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
        data_w,data_b = comm.recv(source=master,tag=11)
        print rank,"received the data in epoch", epoch
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
        print rank,"sending the data in epoch", epoch
        # return slave results
        comm.send((data_w, data_b), dest=master, tag=11)
        
    print "core", rank, "(slave) is done"
      
    
    
    
    
    
    
    