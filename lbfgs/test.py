__author__ = 'billywu'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import matplotlib.pyplot as plt

import tensorflow as tf
import time
from lbfgs_optimizer import lbfgs_optimizer

# Parameters
learning_rate = 0.01
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

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess=tf.Session(config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 1},
                            inter_op_parallelism_threads=8,
                            intra_op_parallelism_threads=8))
sess.run(init)
batch_xs, batch_ys = mnist.train.next_batch(20000)
trainer=lbfgs_optimizer(learning_rate, cost,{x: batch_xs, y: batch_ys},sess,3)


start=time.time()
for i in range(100):
    trainer.feed={x: batch_xs, y: batch_ys}
    s=trainer.minimize()
    print(time.time()-start, sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}),sess.run(cost, feed_dict={x: mnist.test.images, y: mnist.test.labels}),
          sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}),sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}),
          s)