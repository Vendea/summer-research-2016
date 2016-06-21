__author__ = 'billywu'
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
from BFGSoptimizer import BFGSoptimizer
import time

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.ones([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



var=[]
bias={}
bias['b']=b
weight={}
weight['w']=W
data_x, data_y = mnist.train.next_batch(10000)
feed={x:data_x,y_:data_y}
mini=BFGSoptimizer(cross_entropy,feed,[bias,weight],sess)
for tl in [bias,weight]:
    for t in tl:
        var.append(tl[t])

start=time.time()
mini.minimize(cross_entropy,100,0.001)
end=time.time()
print end-start
print sess.run(cross_entropy, feed_dict={x: data_x, y_: data_y})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data_x, y_: data_y}))

