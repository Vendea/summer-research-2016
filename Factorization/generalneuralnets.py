'''
Created on Jun 17, 2016

@author: KatherineMJB
'''
import prime
import tensorflow as tf
import numpy as np

n_layer = 3
n_nodes = 256
nbits = 16
kernel = 0
cost_func = 0
training_sizes = 50000
testing_sizes = 40000
data_x, data_y = prime.generate_data(nbits)
train_x = data_x[0:training_sizes]
test_x = data_x[-(testing_sizes + 1):-1]
train_y = data_y[0:training_sizes]
test_y = data_y[-(testing_sizes + 1):-1]
activation = {0:tf.nn.sigmoid, 1:tf.nn.relu, 2:tf.nn.tanh, 3:tf.nn.softmax}
learning_rate = .0001
training_epochs = 10000
batch_size = 1000

x = tf.placeholder('float', [None, nbits*2])
y = tf.placeholder('float', [None, nbits])

def multilayer_perceptron(x, weights, biases):
    layer = [x]
    for i in range(1,n_layer+1):
        layer.append(activation[kernel](tf.add(tf.matmul(layer[i-1], weights[i]), biases[i])))
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
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(training_sizes/batch_size)
        for i in range(total_batch):
            batch_x=train_x[i*batch_size:(i+1)*batch_size]
            batch_y=train_y[i*batch_size:(i+1)*batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            avg_cost += c / total_batch
        if epoch % 100 == 0:
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


