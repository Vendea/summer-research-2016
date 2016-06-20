'''
Created on Jun 17, 2016

@author: KatherineMJB
'''
import prime
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
from Factorization.7layered40000test import batch_x

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_layer = 3
n_nodes = 256
nbits = 16
kernel = 0
cost_func = 0
training_sizes = 5
testing_sizes = 4
data_x, data_y = prime.generate_data(nbits)
train_x = data_x[0:training_sizes]
test_x = data_x[-(testing_sizes + 1):-1]
train_y = data_y[0:training_sizes]
test_y = data_y[-(testing_sizes + 1):-1]
activation = {0:tf.nn.sigmoid, 1:tf.nn.relu, 2:tf.nn.tanh, 3:tf.nn.softmax}
learning_rate = .0001
training_epochs = 10000
batch_size = 1
num_cores = 4
l_cores = [0,1,2,3]
l_cores = sorted(l_cores)
master = l_cores[0]
slaves = l_cores[1:num_cores]
core_dict = {}
counter = 0
for c in slaves:
    core_dict[c] = counter
    counter += 1

x = tf.placeholder('float', [None, nbits*2])
y = tf.placeholder('float', [None, nbits])

def multilayer_perceptron(x, weights, biases):
    layer = [x]
    for i in range(1,n_layer+2):
        layer.append(activation[kernel](tf.add(tf.matmul(layer[i-1], weights[i-1]), biases[i-1])))
    return layer[-1]

weights = [tf.Variable(tf.random_normal([nbits*2, n_nodes]))]
biases = []
for i in range(1, n_layer):
    biases.append(tf.Variable(tf.random_normal([n_nodes])))
    weights.append(tf.Variable(tf.random_normal([n_nodes, n_nodes])))
weights.append(tf.Variable(tf.random_normal([n_nodes, nbits])))
biases.append(tf.Variable(tf.random_normal([n_nodes])))
biases.append(tf.Variable(tf.random_normal([nbits])))

pred = multilayer_perceptron(x, weights, biases)



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

init = tf.initialize_all_variables()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()

sess=tf.InteractiveSession()
sess.run(init)

# if this core is the master
if rank%num_cores == master:
    data_w = []
    data_b = []
    for w,b in zip(weights, biases):
        data_w.append(w.eval())
        data_b.append(b.eval())

    data = (data_w, data_b)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(training_sizes/batch_size)
        
        for i in l_cores:
            comm.send(data, dest=i, tag=11)
        
        weights_n = []
        biases_n = []
        data_n = []
        
        for i in l_cores:
            data = comm.recv(source=i, tag=11)
            weights_n.append(data[0])
            biases_n.append(data[1])
            
        avg_weight = np.average(weights_n, axes=0)
        avg_bias = np.average(biases_n, axes=0)
                
        if epoch % 100 == 0:
            for w,t in zip(avg_weight, weights):
                sess.run(t.assign(w))
            for b,t in zip(avg_bias, biases):
                sess.run(t.assign(b))
                
            test_cost = sess.run(cost,{x:test_x, y:test_y})
            print "epoch",epoch,"test_cost:", test_cost
            train_cost = sess.run(cost, {x:train_x, y:train_y})
            print "epoch",epoch,"train_cost:", train_cost
            train_pred = sess.run(pred, {x:train_x, y: train_y})
                
            correct = 0
            for tp, ty in zip(train_pred, train_y):
                valid = True
                for tpe, tye in zip(tp, ty):
                    if tpe > 0 and tye == 0:
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
                    if tpe > 0 and tye == 0:
                        valid = False
                        break
                    elif tpe <= 0 and tye ==1:
                        valid = False
                        break
                if valid:
                    correct = correct +1
            print "epoch", epoch, "testing accuracy:", (correct + 0.0)/testing_sizes
    print "master is done"

#if one of the slave cores
else:
    local_size = training_sizes / (num_cores-1)
    p = core_dict[rank]
    local_train_x = train_x[p*local_size:(p+1)*local_size]
    local_train_y = train_y[p*local_size:(p+1)*local_size]
    
    for epoch in range(training_epochs):
        data = comm.recv(source=master,tag=11)
        for w,t in zip(data[0], weights):
            sess.run(t.assign(w))
        for b,t in zip(data[1], biases):
            sess.run(t.assign(b))
            
        for i in range(local_size/batch_size):
            batch_x=train_x[i*batch_size:(i+1)*batch_size]
            batch_y=train_y[i*batch_size:(i+1)*batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        data_w = []
        data_b = []
        for w,b in zip(weights, biases):
            data_w.append(w.eval())
            data_b.append(b.eval()) 
        comm.send((data_w, data_b), dest=master, tag=11)
        
    print "core", rank, "(slave) is done"
      
    
    
    
    
    
    
    