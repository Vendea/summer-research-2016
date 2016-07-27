import time
from sys import path
from os import getcwd

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpi4py import MPI

p = getcwd()[0:getcwd().rfind("/")]+"/SGD"
path.append(p)

p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)
import Logger
logfile = Logger.DataLogger("MNIST_SGD","Epoch,time,train_accuaracy,test_accuaracy,train_cost,test_cost")

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
training_size = len(data_x)
param=[]

for t in tf.trainable_variables():
    param.append(t.eval(session=sess))
if rank==0:
    server=ParamServer(param,comm)
    while True:
        core,data=server.next_request([x for x in range(1,size)])
        server.handle_request(core,data)
        
else:
    data=data_x[training_size/(size-1)*(rank-1):training_size/(size-1)*(rank)],data_y[training_size/(size-1)*(rank-1):training_size/(size-1)*(rank)]
    worker=DPSGD(param,data,batch_size,comm,train_step,sess,x,y,cost,rank,0,accuracy,{x: mnist.test.images, y:mnist.test.labels})
    start=time.time()
    while True:
        for i in range(100):
            worker.optimize()
            worker.publish()
            train=sess.run(accuracy,{x:data[0],y:data[1]})
            test= sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels})
            train_cost=sess.run(cost,{x:data[0],y:data[1]})
            test_cost= sess.run(cost,{x:mnist.test.images,y:mnist.test.labels})
            
            logfile.writeData((i,time.time()-start, train, test,train_cost,test_cost))