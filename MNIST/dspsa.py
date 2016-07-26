import time
from sys import path
from os import getcwd

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpi4py import MPI

p = getcwd()[0:getcwd().rfind("/")]+"/SGD"
path.append(p)

from ParamServer import ParamServer
from ModelReplica import DPSGD




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100

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

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch the graph
config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 0},
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
sess=tf.Session(config=config)
sess.run(init)
data_x, data_y = mnist.train.images,mnist.train.labels
feed={x:data_x,y:data_y}
mini=SPSA(cost,feed,sess)

if rank==0:
    start=time.time()
    for n in range(1000):
        orig=mini.var
        g0=mini.getGrad(cost,n)
        f,update0=mini.minimize(g,orig,n)
        a0=sess.run(accuracy,feed)
        f0=sess.run(cost,feed)
        update1=comm.recv(source=2,tag=11)
        mini.set_var(update1)
        a1=sess.run(accuracy,feed)
        f1=sess.run(cost,feed)
        print time.time()-start,f1,f0,a1,a0
        if a1>a0:
            update=comm.bcast(update1,root=0)
        else:
            update=comm.bcast(update0,root=0)
        mini.set_var(update)
        if n%10==0:
            print sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels})
else:
    for n in range(1000):
        orig=mini.var
        g0=mini.getGrad(cost,n)
        f,update=mini.minimize(g,orig,n)
        comm.send(update,dest=0,tag=11)
        update=comm.bcast(None,root=0)
        mini.set_var(update)


