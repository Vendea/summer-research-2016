__author__ = 'billywu'
from mpi4py import MPI
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import time
import tensorflow as tf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ncore=3

epoch=100

if rank!=0:
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    sess = tf.InteractiveSession()
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(init)

if rank == 0:
    start=time.time()
    w=np.zeros([784,10])
    bias=np.zeros([10])
    data=(w,bias)
    iter=0
    ws=[]
    bs=[]
    while iter<epoch:
        print "epoch", iter
        for i in range(ncore):
            comm.send(data,dest=i,tag=11)
        for i in range(ncore):
            w,b=comm.recv(source=i,tag=11)
            w=np.array(w)
            b=np.array(b)
            ws.append(w)
            bs.append(b)
        w_avg=np.zeros(w.shape)
        for w in ws:
            w_avg=w_avg+w
        w=w/ncore
        b_avg=np.zeros(b.shape)
        for b in bs:
            b_avg=b_avg+b
        b=b/ncore
        data1=(w,b)
        iter=iter+1
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(data1[0])
    b = tf.Variable(data1[1])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    sess = tf.InteractiveSession()
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(init)
    print "master accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    end=time.time()
    print "Core0 done, process time:", end-start, 'seconds'
elif rank !=0:
    iter=0
    while iter<epoch:
        batch_xs, batch_ys = mnist.train.next_batch(1200)
        weight,bias=comm.recv(source=0,tag=11)
        assign_opW = W.assign(weight)
        assign_opb = b.assign(bias)
        batch_xs, batch_ys = batch_xs[(rank-1)*1200/ncore:rank*1200/ncore+1], batch_ys[(rank-1)*1200/ncore:rank*1200/ncore+1]
        for i in range(10):
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        weight=W.eval()
        bias=b.eval()
        data=(weight,bias)
        #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        comm.send(data,dest=0,tag=11)
        iter=iter+1
    print "Core",rank," done"


