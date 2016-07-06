
#Gets the imports for the system
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time
from DistributedSPSA import SPSA
from mpi4py import MPI
import sys
from sys import path
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)
import Logger as l

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  # gets the Data set
args = len(sys.argv)                                           # gets the number of command line arguments used to differeniate between workers and masters
#Sets up variables to be used for communication and logging
comm_children = None                                           
comm_parent   = None
correct_prediction,accuracy = None,None
train_cost,train_accuracy = [],[]
test_accuracy = []
# Network Parameters
learning_rate = 0.001
training_epochs = 600
batch_size   = 100
n_layer      = 2
n_nodes      = 256
n_input      = 784 # MNIST data input (img shape: 28*28)
n_classes    = 10 # MNIST total classes (0-9 digits)
n_workers    = 2 # number of workers each master has to help calculate gradients 
gen_size     = 1 # number of networks to save each generation 
train_size   = 4000# number of elements in the train and testing netowrks 
start_worker = "worker" # random keyword used so signify a worker thread
#random stuff
weights = [] # list of wieghts
biases = [] # list of biases
pred = None # varibale used to store the last layer of network
# tf Graph input
x = tf.placeholder("float", [None, n_input]) # variable used to store the input
y = tf.placeholder("float", [None, n_classes])# used to store true value

if args == 1:
    # starts master theads and spawns the relavent workers and creates communicators from the masters to workers
    comm_global = MPI.COMM_WORLD
    rank = comm_global.Get_rank()
    size = comm_global.Get_size()
    comm_children = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[sys.argv[0], start_worker],
        maxprocs=n_workers*size)
    start=n_workers*rank
    end = start+n_workers -1
    if rank == 0:
        # on root starts a logger to save results 
        f = l.DataLogger("SPSA"
            ,n_layer
            ,n_nodes
            ,header="Epoch,(cost,accuracy),Computation_Time,Train_Accuracy,Test_Accuracy",
            testing=True)
else:
    # starts comminicator for worker
    comm_global = MPI.COMM_WORLD
    comm_parent = MPI.Comm.Get_parent()
    rank = comm_parent.Get_rank()
    size = comm_parent.Get_size()
# Constructing the hidden and output layers
def multilayer_perceptron():
    prevlayer = x
    for w,b in zip(weights,biases):
        prevlayer = tf.nn.relu(tf.add(tf.matmul(prevlayer, w), b))
    return prevlayer

def build_network_master():
    global pred,accuracy
    # Builds master and sends its description to all other nodes so they all start at the same intial point
    # Defining and initializing the trainable Variables
    weights.append(tf.Variable(tf.random_normal([n_input, n_nodes])))
    for i in range(1, n_layer):
        biases.append(tf.Variable(tf.random_normal([n_nodes])))
        weights.append(tf.Variable(tf.random_normal([n_nodes, n_nodes])))
    weights.append(tf.Variable(tf.random_normal([n_nodes, n_classes])))
    biases.append(tf.Variable(tf.random_normal([n_nodes])))
    biases.append(tf.Variable(tf.random_normal([n_classes])))
    # Defining the output
    data_w = []
    data_b = []
    init = tf.initialize_all_variables()
    # Launch the graph
    sess.run(init)
    for w,b in zip(weights, biases):
        data_w.append(w.eval())
        data_b.append(b.eval())
    #sends data to all masters that are not root and to roots children
    data = zip(data_w, data_b)
    for i in range(1,size):
        comm_global.send(data, dest=i,tag=11)
    for i in range(start,end):
        comm_children.send(data, dest=i,tag=11)
    # builds the operations and the accuaracy functions 
    pred = multilayer_perceptron()
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   
def build_network_slaves():
    # buids network for masters that are not root and send s the data to all of there children
    global pred,accuracy
    data = None
    data = comm_global.recv(source=0,tag =11)
    for i in range(start,end):
        comm_children.send(data, dest=i,tag=11)
    # builds the variables 
    # Defining and initializing the trainable Variables
    for w,b in data:
        biases.append(tf.Variable(b))
        weights.append(tf.Variable(w))
    # Defining the output
    pred = multilayer_perceptron()
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    init = tf.initialize_all_variables()
    # Launch the graph
    sess.run(init)

def build_network_children():
    global pred
    data = None
    data = comm_parent.recv(source=rank/n_workers,tag =11)
    # Defining and initializing the trainable Variables
    for w,b in data:
        biases.append(tf.Variable(b))
        weights.append(tf.Variable(w))
    # Defining the output
    pred = multilayer_perceptron()
    init = tf.initialize_all_variables()
    # Launch the graph
    sess.run(init)

def update_weights_slaves(c,accu,update):
    comm_global.send([c,accu,update,], dest=0, tag=11)
    data = comm_global.recv(source=0, tag=11)
    mini.set_var(data)
    update_weights_parent(data)

def update_weights_master(c,accu,update):
    data,ls = [],[]
    del train_cost[:]
    for i in range(1,size):
            ls.append(comm_global.recv(source=i, tag=11))
    update_cost,update_accu,update_values = zip(*ls)
    update_cost = list(update_cost)
    update_accu = list(update_accu)
    update_values = list(update_values)
    update_cost.append(c)
    update_accu.append(accu)
    update_values.append(update)
    while len(data) < gen_size:
        i = update_accu.index(max(update_accu))
        train_cost.append(
            (update_cost.pop(i),update_accu.pop(i))
            )
        data.append(update_values.pop(i))
    for i in range(size):
        if i != 0:
            comm_global.send(data[i%gen_size],dest=i, tag=11)
        else:
            mini.set_var(data[i%gen_size])
            update_weights_parent(data[i%gen_size])

def update_weights_children(g):
    comm_parent.send([g], dest=rank/n_workers, tag=11)
    mini.set_var(comm_parent.recv(source=rank/n_workers, tag=11))

def update_weights_parent(data):
    for i in range(start,end):
            comm_children.send(data,dest=i, tag=11)

with tf.Session() as sess:
    if rank == 0 and comm_parent == None:
        build_network_master()
    elif comm_parent == None:
        build_network_slaves()
    else:
        build_network_children()

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    # Initializing the variables
   

    data_x, data_y = mnist.train.next_batch(train_size)
    test_x, test_y = mnist.test.next_batch(train_size)
    feed={x:data_x,y:data_y}
    mini=SPSA(cost,feed,biases+weights,sess)

    for ep in range(training_epochs):
        timer = 0
        start_time = time.time()
        t = time.time()
        g = mini.nabla(cost,ep)
        if rank == 0 and args == 1:
                print( "gradient computation",time.time()- t)
        if comm_parent != None:
            update_weights_children(g)
        else:
            dv = []
            for i in range(start,end):
                dv.append(comm_children.recv(source=i,tag=11))
            orginal,update = mini.minimize(dv,ep)
            t = time.time()
            c,accu=sess.run([cost,accuracy],feed)
            if rank == 0 and args == 1:
                print("cost and accuarcy computation",time.time()- t)
            if rank != 0:
                update_weights_slaves(c,accu,update)
            else:
                t = time.time()
                update_weights_master(c,accu,update)
                if rank == 0 and args == 1:
                    print("update step",time.time()- t)
        end_time = time.time()
        if args == 1 and rank == 0:
            t = time.time()
            _,train_accuracy= sess.run([cost,accuracy],{x:data_x,y:data_y})
            _,test_accuracy = sess.run([cost,accuracy],{x:test_x,y:test_y})
            if rank == 0 and args == 1:
                print( "final accuarcy check",time.time()- t)
            f.writeData((ep,
                        train_cost,
                        end_time-start_time,
                        train_accuracy,
                        test_accuracy))
            timer += end_time-start_time
            print("time taken for epoch"+str(ep)+":"+str(end_time-start_time))