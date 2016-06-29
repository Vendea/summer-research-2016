'''
Created on Jun 17, 2016
@author: Katherine Beine, Tray Hurley, Eyob Tsegaye, Mingsheng Wu

This is a distributed implementation of a neuralnets framework using tensorflow used to factorize large integers.
'''
#Imports needed packages 
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
import logging
import sys
import pickle
from BFGS_NL import BFGSoptimizer 
#Starts the system logger to print relavent information 
logging.basicConfig(level=logging.INFO)
logger  = logging.getLogger(__name__)
''' 
===============================================================================================
                                Command Line Arguments  
===============================================================================================
    Optimizer 0
    Number of workers 1
    size of the batchs 2
    cost function 3
    activation function 4
    accuarcy checking method 5
    number of elemetns in train set 6
    number of elements in test set 7
    trainning epochs 8
    network[layers,nodes] 9
    loc of data 10
    State key either MASTER or SLAVE 11
    rank passed to master only 12
'''
#gets the command line arguments and converts them to the correct types and generates the command line for children  
args    = sys.argv[1:]
temp    = []

for x in args:
    temp.append(eval(x))
spawn = args[0:11]+["\"SLAVE\""]+[args[12]]
args = temp

#loads the date from saved pickle objects 
train   = pickle.load(open(args[10][0],"r"))
test    = pickle.load(open(args[10][1],"r"))
'''===============================================================================================
    Preprocessing and Configurations
   ===============================================================================================
'''
#   1. specifying the number of cores allocated
num_cores = args[1]
# Configuring IPC:
#   2. initializing MPI
status = args[11]
if status== "MASTER":    
    rank = args[12]
    f = open('stuff'+str(rank)+".txt", 'w')
#   3. specifying the core identification numbers
l_cores = [x for x in range(num_cores-1)]
#   3. determining the master/slave model
counter = 0
master  = 0 
core_dict = {}
for c in l_cores:
    core_dict[c] = counter
    counter += 1
#adds the master to the list fo object to work on 
core_dict[num_cores-1] = counter +1 
# starts the worker threads 
if status == "MASTER" and num_cores != 1:
    comm = MPI.COMM_WORLD.Spawn(sys.executable,args=["GenNet_MNIST.py"]+spawn,maxprocs=num_cores-1)
    rank = num_cores-1
    for i in l_cores:
        #makes sure the workers are started so that all threads can communicate
        data = comm.recv(source=i, tag=11)
elif  status == "SLAVE":
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    size = comm.Get_size()
    # tells the master that it is started so that they can communication
    comm.send("Started", dest=master, tag=11)
# Configuring NN Structure:
network = args[9]

if type(network) is list:
    saved_Graph = False
    #   1. predefined number of layers
    n_layer = int(args[9][0])
    #   2. predefined number of nodes per hidden layer
    n_nodes = int(args[9][1])
else:
    saved_Graph = True
    content = None
    with open(network, 'r') as content_file:
        content = content_file.read()
    graph_def = content
#   3. cost function select
#           (used to study how the different cost function
#            can lead to different convergence properties):
#       0: RMSE
#       1: Cross Entropy
#       2: Absolute factor differences
#       3: bitwise differences
cost_func = args[3]
#   4. kernel select:
#       0: sigmoid
#       1: relu
#       2: tanh
#       3. softmax
kernel     = args[4]
activation = {0:tf.nn.sigmoid, 1:tf.nn.relu, 2:tf.nn.tanh, 3:tf.nn.softmax}
#   5. Optimizer select:
#       0: sigmoid
#       1: relu
#       2: tanh
#       3. softmax
opt = {
"GD":tf.train.GradientDescentOptimizer,
"ADSGD":tf.train.GradientDescentOptimizer,
"BFGS":BFGSoptimizer,
"SPSA":None,
"ADAM":tf.train.AdamOptimizer,
"DDSGD":tf.train.GradientDescentOptimizer,
"SBFGS":None
}
opt_key = args[0]
#Starts the server on each thread 
jobs = ["localhost:22"+str(args[12])+str(i) for i in range(num_cores)] 
cluster = tf.train.ClusterSpec({"worker": jobs}) 
server = tf.train.Server(cluster, job_name="worker", task_index=rank)
#if thread is only used for computaion then thead locks untill master is done with it 
if (opt_key == "GD" or opt_key == "BFGS" or opt_key == "SPSA" or opt_key == "ADAM") and status == "SLAVE":
    while comm.recv(source=master,tag=11) != "END":
        print("Worker Server Running...")
    print("Server Ended...")
    
else:
    #   6. Accuarcy check select:
    #       0: sigmoid
    #       1: relu
    #       2: tanh
    #       3. softmax
    acu = args[5]
    # Configuring Learning Properties:
    #   1. learning rate
    learning_rate = .0001
    #   2. learning epochs
    training_epochs = int(args[8])
    #   3. mini-batch sizes
    batch_size = args[2]
    batching   = batch_size != 0 
    #   4. evaluating frequency
    display_step=1
    # Configuring Problem Specific Data Setup
    #   2. training size
    training_sizes = args[7]
    #   3. testing size
    testing_sizes = args[8]
    #   4. obtaining the data
    train_data_x, train_data_y =  train[0:training_sizes]
    test_data_x, test_data_y =  test[0:testing_sizes]
    #   5. splitting the training and testing data
    train_x = train_data_x
    test_x = test_data_x
    train_y = train_data_y
    test_y = test_data_y
    # assign the size of the input and output nodes
    input_size = len(train_x[0])
    out_size = len(train_y[0])
    device_counter = 0  
    logger.info("Data preprocessing done in core"+str(rank))
    '''===============================================================================================
        Neural Network Model Tensorflow Replicas Construction
       ===============================================================================================
    '''
    # Constructing the hidden and output layers
    def multilayer_perceptron(x, weights, biases):
        layer = [x]
        for i in range(1,n_layer+2):
            layer.append(activation[kernel](tf.add(tf.matmul(layer[i-1], weights[i-1]), biases[i-1])))
        return layer[-1]

    def random_device_placment(n):
        global  device_counter,opt_key
        d = "/job:worker/task:"+str(rank)
        if not (opt_key =="ADSGD" or opt_key =="DDSGD" or opt_key == "SBFGS"):
            d = "/job:worker/task:"+str(device_counter)
            device_counter += 1 
            if device_counter == num_cores:
                device_counter = 0 
        else:
            "/job:worker/task:"+str(rank)
        return d 
    # Specifying the inputs into the graph
    with tf.device(random_device_placment):
        x = tf.placeholder('float', [None, input_size])
        y = tf.placeholder('float', [None, out_size])
        pred = None
        if not saved_Graph:
            # Defining and initializing the trainable Variables
            weights = [tf.Variable(tf.random_normal([input_size, n_nodes]))]
            biases = []
            for i in range(1, n_layer):
                biases.append(tf.Variable(tf.random_normal([n_nodes])))
                weights.append(tf.Variable(tf.random_normal([n_nodes, n_nodes])))

            weights.append(tf.Variable(tf.random_normal([n_nodes, out_size])))
            biases.append(tf.Variable(tf.random_normal([n_nodes])))
            biases.append(tf.Variable(tf.random_normal([out_size])))
            # Defining the output
            pred = multilayer_perceptron(x, weights, biases)
        else:
            tf.import_graph_def(graph_def, input_map={"input":x,"output":y},name="")
            pred = tf.get_default_graph().get_operation_by_name("pred")

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
        # Defining the accuarcy method based on the function select
        if acu == 0:
            correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif acu == 1:
            correct_prediction = tf.equal(y,pred)
            correct_prediction = tf.reduce_all(correct_prediction)# needs to become a soft margin
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Optimizer
        if (opt_key != "BFGS"):
            optimizer = opt[opt_key](learning_rate=learning_rate).minimize(cost)
        # Graph initialization
        init = tf.initialize_all_variables()
    with tf.Session(server.target) as sess:
        sess.run(init)
        logger.info("Network initialization done in core"+str(rank))
        '''===============================================================================================
            Training and Model evaluation
           ===============================================================================================
        '''
        data_w = []
        data_b = []
        # Master core code
        if status == "MASTER":
            timer = 0
            print "master started..."
            # Constructing the initial message
            if opt_key =="ADSGD" or opt_key =="DDSGD" or opt_key == "SBFGS":
                
                for w,b in zip(weights, biases):
                    data_w.append(w.eval())
                    data_b.append(b.eval())
                data = (data_w, data_b)
                print "master core started..."
        local_size = training_sizes / (num_cores)   
        p = core_dict[rank]
        if status == "MASTER":
            p = core_dict[num_cores - 1]
        # Initializing local data shard
        local_train_x = train_x[p*local_size:(p+1)*local_size]
        local_train_y = train_y[p*local_size:(p+1)*local_size]
        if (opt_key == "BFGS"):
            batch_x=local_train_x
            batch_y=local_train_y
            optimizer = opt[opt_key](cost=cost,feed={x: batch_x, y: batch_y},sess=sess)

        if status == "MASTER":
                # distributing tasks to the following slave cores
                if num_cores != 1 and (opt_key =="DDSGD" or opt_key == "SBFGS"):
                    for i in l_cores:
                        comm.send(data, dest=i, tag=11)
        else:
            if (opt_key =="DDSGD" or opt_key == "SBFGS"):
                    data_w,data_b = comm.recv(source=master,tag=11)
                    # apply the variables
                    for w,t in zip(data_w, weights):
                        sess.run(t.assign(w))
                    for b,t in zip(data_b, biases):
                        sess.run(t.assign(b))
        for epoch in range(training_epochs):
            if status == "MASTER":
                # distributing tasks to the following slave cores
                if num_cores != 1 and (opt_key =="ADSGD"):
                    for i in l_cores:
                        comm.send(data, dest=i, tag=11)
            else:
                # waiting on weight initialization
                #print("recive weights")
                if (opt_key =="ADSGD"):
                    data_w,data_b = comm.recv(source=master,tag=11)
                    # apply the variables
                    for w,t in zip(data_w, weights):
                        sess.run(t.assign(w))
                    for b,t in zip(data_b, biases):
                        sess.run(t.assign(b))
            # batching training    
            if batching:
                start_time = time.time()
                for i in range(training_sizes/batch_size):
                    batch_x=local_train_x[i*batch_size:(i+1)*batch_size]
                    batch_y=local_train_y[i*batch_size:(i+1)*batch_size]
                    if opt_key != "BFGS":
                        _,c,train_accuracy = sess.run([optimizer, cost,accuracy], feed_dict={x: batch_x, y: batch_y})
                    else:
                        optimizer.minimize()
                        c,train_accuracy = sess.run([cost,accuracy], feed_dict={x: batch_x, y: batch_y})
            else:
                start_time = time.time()
                batch_x=local_train_x
                batch_y=local_train_y
                if opt_key != "BFGS":
                    _, c,train_accuracy = sess.run([optimizer, cost,accuracy], feed_dict={x: batch_x, y: batch_y})
                else:
                    optimizer.minimize()
                    c,train_accuracy = sess.run([cost,accuracy], feed_dict={x: batch_x, y: batch_y})
            if status == "MASTER":
                f.write("epoch "+ str(epoch)+" core "+str(rank)+" training accuracy: "+str(train_accuracy)+"\n")
                 # collecting the weights and biases obtained from each slave cores
                weights_n = []
                biases_n = []
                data_n = []
                if num_cores != 1 and opt_key =="ADSGD":
                    data_w = []
                    data_b = []
                    for w,b in zip(weights, biases):
                        data_w.append(w.eval())
                        data_b.append(b.eval())
                    for i in l_cores:
                 #           print("updates")
                        data = comm.recv(source=i, tag=11)
                        weights_n.append(data[0])
                        biases_n.append(data[1])
                    
                    weights_n.append(data_w)
                    biases_n.append(data_b)
                     # averaging the changes
                    avg_weight = np.average(weights_n, axis=0)
                    avg_bias = np.average(biases_n, axis=0)
                    data=(avg_weight,avg_bias)
                    # evaluating the training progress every display_step epochs
                    if (epoch % display_step) == 0:
                        for w,t in zip(avg_weight, weights):
                            sess.run(t.assign(w))
                        for b,t in zip(avg_bias, biases):
                            sess.run(t.assign(b))
                    # evaluating on the cost function
                test_cost,train_accuracy = sess.run([cost,accuracy],{x:test_x, y:test_y})
                f.write("epoch "+str(epoch)+" test_cost: "+ str(test_cost)+"\n")
                train_cost, test_accuracy = sess.run([cost,accuracy], {x:train_x, y:train_y})
                f.write( "epoch "+str(epoch)+" train_cost: "+str(train_cost)+"\n")
                # evaluating on the actual accuracy
                f.wrtie("epoch "+ str(epoch)+ " training accuracy: "+str(train_accuracy)+"\n")
                f.write("epoch "+ str(epoch)+ " testing accuracy: "+ str(test_accuracy)+"\n")
                end_time = time.time()
                timer += end_time-start_time
                if (epoch % display_step) == 0:
                    logger.info("time taken for epoch"+str(epoch)+":"+str(end_time-start_time)+"\n")
                f.write("avg time per epoch:"+str(timer/training_epochs)+"\n")
            else:
                # return slave results
                #print("send updates")
                if opt_key =="ADSGD":
                    comm.send((data_w, data_b), dest=master, tag=11)
            if status == "MASTER" and num_cores != 1:
                for i in l_cores:
                                comm.send("END", dest=i, tag=11)
            elif status == "SLAVE" and num_cores != 1:
                while comm.recv(source=master,tag=11) != "END":
                    print("Gracefully Ending Worker...")
        print("Done"+str(rank))
   
            