'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import time
from Multi_try_Metropolis import MCMC
import matplotlib.pyplot as plt
from mpi4py import MPI
import random
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = 2
master = (rank/n_workers)*n_workers

local_comm = comm.Split(rank/n_workers, rank)
if rank % n_workers == 0:
    royal_comm = comm.Split(0, rank)
else:
    royal_comm = comm.Split(MPI.UNDEFINED, rank)

generations = 10

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

def r_squared(x, y):
    ux = float(sum(x))/len(x)
    uy = float(sum(y))/len(y)
    num = sum([(x[i]-ux)*(y[i]-uy) for i in range(0, len(x))])
    den1 = sum([(xi-ux)*(xi-ux) for xi in x])
    den2 = sum([(yi-uy)*(yi-uy) for yi in y])
    return float(num*num)/(den1*den2)

def performance(stdev):
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

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

    # Launch the graph
    config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 1},
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    sess=tf.InteractiveSession(config=config)
    sess=tf.Session()
    sess.run(init)


    data_x, data_y = mnist.train.next_batch(10000)
    feed={x:data_x,y:data_y}
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #mini=MCMC(cost,feed,sess)
    mini=MCMC(accuracy,{x: mnist.test.images, y:mnist.test.labels},sess,root=0,comm=local_comm)
    costs = []
    costs.append(mini.prev_cost)
    timestamps = [0]
    start = time.time()
    for ep in range(5):
        if rank == 0:
            #print ep
        mini.optimize(stdev=stdev)
        timestamps.append(time.time() - start)
        costs.append(mini.prev_cost)
        #if master:
        #    print time.time()-start, mini.prev_cost
        #if time.time()-start > 300:
        #    break
        #print sess.run(cost, feed)
        #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print "Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels},session=sess)
    if local_comm.Get_rank() == 0:
        epochs = [i for i in range(len(costs))]
        bfl = np.poly1d(np.polyfit(epochs, costs, 1))
        return (stdev, r_squared(epochs, costs), bfl(len(costs)-1)-bfl(0))
    else:
        return None

def median(a):
    l = len(a)
    if l % 2 == 0:
        return (sorted(l)[len(l)/2] + sorted(l)[len(l)/2 - 1]) / 2.0
    else:
        return sorted(l)[len(l)/2]

def sort_into_lohi(perfs, r2_med, diff_med):
    lohi = {'hh':[], 'hl':[], 'lh':[], 'll':[]}
    for perf in perfs:
        if perf[1] > r2_med:
            if perf[2] > diff_med:
                lohi['hh'].append(perf)
            else:
                lohi['hl'].append(perf)
        else:
            if perf[2] > diff_med:
                lohi['lh'].append(perf)
            else:
                lohi['ll'].append(perf)
    return lohi

def evaluate(perf):
    if perf[2] > 0.01:
        if perf[1] > 0.4:
            return 'hh'
        else:
            return 'lh'
    else:
        if perf[1] > 0.4:
            return 'hl'
        else:
            return 'll'

def optimal_perf(perfs):
    r2s = []
    diffs = []
    for perf in perfs:
        r2s.append(perf[1])
        diffs.append(perf[2])
    #r2_med = median(r2s)
    #diff_med = median(diffs)
    #lohi = sort_into_lohi(perfs, r2_med, diff_med)
    lohi = {'hh': [], 'hl': [], 'lh': [], 'll': []}
    for perf in perfs:
        lohi[evaluate(perf)].append(perf)
    if lohi['hh'] != []:
        selection = lohi['hh']
    elif lohi['lh'] != []:
        selection = lohi['lh']
    elif lohi['hl'] != []:
        selection = lohi['hl']
    else:
        selection = lohi['ll']

    if len(selection) == 1:
        return selection[0]
    else:
        diffs = [perf[2] for perf in selection]
        max_diff = max(diffs)
        if diffs.count(max_diff) == 1:
            return perfs[diffs.index(max_diff)]
        else:
            r2s = [perf[1] for perf in selection]
            return perfs[r2s.index(max(r2s))]

def gen_stdev(perf):
    evaluation = evaluate(perf)
    if evaluation == 'hh':
        #basically stay where you are
        return random.gauss(perf[0], 0.001)
    elif evaluation == 'lh':
        #slightly decrease
        return random.gauss(perf[0]-0.005, 0.002)
    elif evaluation == 'hl':
        #increase
        return random.gauss(perf[0]+0.01, 0.003)
    else:
        #decrease
        return random.gauss(perf[0]-0.01, 0.003)


def emporer_step(stdev):
    perfs = royal_comm.gather(performance(stdev), root=0)
    perf = optimal_perf(perfs)
    with open('file.txt', 'a') as f:
        f.write(str(perf[0])+'\n')
    stdevs = [gen_stdev(perf) for _ in range(1, royal_comm.Get_size())]
    stdevs.append(perf[0])
    stdev = royal_comm.scatter(stdevs, root=0)
    local_comm.bcast(stdev, root=0)
    return stdev


def master_step(stdev):
    royal_comm.gather(performance(stdev), root=0)
    stdev = royal_comm.scatter(None, root=0)
    local_comm.bcast(stdev, root=0)
    return stdev

def worker_step(stdev):
    perf = performance(stdev)
    stdev = local_comm.bcast(None, root=0)
    return stdev

if local_comm.Get_rank() == 0:
    stdev = random.random()
    local_comm.bcast(stdev, root=0)
else:
    stdev = local_comm.bcast(None, root=0)

if rank == 0:
    for i in range(generations):
        #print "rank 0 (emporer), gen", i, ": ", stdev
        stdev = emporer_step(stdev)
    #print "rank 0 (emporer) end:", stdev
if local_comm.Get_rank() == 0:
    for i in range(generations):
        #print "rank", rank, "(master), gen", i, ": ", stdev
        stdev = master_step(stdev)
    #print "rank", rank, "(master) end:", stdev
else:
    for i in range(generations):
        #print "rank", rank, "(worker), gen", i, ": ", stdev
        stdev = worker_step(stdev)
    #print "rank", rank, "(worker) end:", stdev
