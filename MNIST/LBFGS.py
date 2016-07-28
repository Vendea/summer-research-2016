import time
from sys import path
from os import getcwd

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpi4py import MPI

p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)



import Logger
logfile = Logger.DataLogger("MNIST_LBFGS","Epoch,time,train_accuaracy,test_accuaracy,train_cost,test_cost")
p = getcwd()[0:getcwd().rfind("/")]+"/lbfgs"
path.append(p)

from lbfgs_optimizer import lbfgs_optimizer
from Opserver import Opserver



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
tx,ty = mnist.train.images[0:30],mnist.train.labels[0:30]
train_size =  len(tx)
bsize=train_size

if rank==0:
    trainer=lbfgs_optimizer(0.0001, cost,[],sess,1,comm,size,rank)
    for b in range(5):
        data_x=tx[bsize*b:bsize*(b+1)]
        data_y=ty[bsize*b:bsize*(b+1)]
        trainer.update(data_x,data_y,x,y)
        start=time.time()
        for i in range(40):
            c = trainer.minimize()
            train=sess.run(accuracy,{x:tx,y:ty})
            test= sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels})
            train_cost=c
            test_cost= sess.run(cost,{x:mnist.test.images,y:mnist.test.labels})
            #f=trainer.functionEval
            #g=trainer.gradientEval
            #i=trainer.innerEval
            #print i, f, g, train, test,train_cost,test_cost
            logfile.writeData((i,time.time()-start, train, test,train_cost,test_cost))
else:
    opServer=Opserver(0.0001, cost,[],sess,comm,size,rank,0,x,y)
    opServer.run()


