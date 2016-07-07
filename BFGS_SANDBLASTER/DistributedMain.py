__author__ = 'billywu'

import time


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpi4py import MPI

from SandblasterMasterOptimizer import BFGSoptimizer
from OperationServer import SandblasterOpServer
from sys import path
import sys
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)
from Logger import DataLogger


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

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

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch the graph
config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 1},
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
sess=tf.Session(config=config)
sess.run(init)
print sys.argv[1]


data_x, data_y = mnist.train.next_batch(10000)
global_feed={x:data_x,y:data_y}

if rank==0:
    logger = DataLogger("BFGS"+sys.argv[1]
                    ,2
                    ,256
                    ,header="Step_Time,Total_Time,Train_Accuracy,Test_Accuracy")
    feed={x:data_x[0:len(data_x)/size],y:data_y[0:len(data_x)/size]}
    mini=BFGSoptimizer(cost,feed,sess,rank,comm)
    start=time.time();prev=0
    for ep in range(100):
        mini.minimize(alpha=0.0001)
        #test_c=cost.eval({x: mnist.test.images, y:mnist.test.labels},session=sess)
        #train_c=cost.eval({x: data_x, y:data_y},session=sess)
        test_acc=accuracy.eval({x: mnist.test.images, y:mnist.test.labels},session=sess)
        train_acc=accuracy.eval({x: data_x, y:data_y},session=sess)
        now=time.time()
	prev= prev+now-start
        logger.writeData((now-start,prev, train_acc,test_acc))
    comm.scatter(["KILL" for x in range(comm.Get_size())],root=0)
    print "Average Gradient Computation Time:", mini.get_average_grad_time()
    print "Core 0 finished."
else:
    feed={x:data_x[len(data_x)/size*rank:len(data_x)/size*(rank+1)],y:data_y[len(data_x)/size*rank:len(data_x)/size*(rank+1)]}
    Operator=SandblasterOpServer(rank, feed, sess, cost,comm)
    total_time=0
    while (True):
        data="None"
        data=comm.scatter(["GP" for x in range(comm.Get_size())],root=0)
        try:
            if data=="None":
                continue
            elif data=="KILL":
                break
        except:
            start=time.time()
            g=Operator.Compute_Gradient(data)
            c=Operator.Compute_Cost()
            new_data = comm.gather((g,c),root=0)
            end=time.time()
            total_time=total_time+end-start
    print "Core,", rank, "Computation Cost:", total_time
    print "Core", rank, "finished."

