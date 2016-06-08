import random as rand
import math as m
import tensorflow as tf

# set up arrays of integers and sums for training and testing
a = [m.floor(rand.random()*2**16) for i in range(0, 2**16)]
b = [m.floor(rand.random()*2**16) for i in range(0, 2**16)]
trainingSums = [v+u for v in a for u in b]
testingSums = [2**17+v+u for v in a for u in b]

traindata = []
testdata = []

def dconvert(x):
    rval = convert(x,2)
    pad = [0 for i in range(16-len(rval))]
    return pad + rval
  
def convert(m,n):
    if(m<n):
        return [int(m)]
    else:
        return convert(int(m/n),n) + [int(m%n)]

for i in range(0,250):
    traindata.append(dconvert(a[i]) + dconvert(b[i]))
    trainingSums[i] = dconvert(trainingSums[i])
    testdata.append(dconvert(a[i]+2**16) + dconvert(b[i]+2**16))
    testingSums[i] = dconvert(testingSums[i])

x = tf.placeholder(tf.float32, [None,20])#makes input nodes

#this section makes the first hidden layer
W1 = tf.Variable(tf.zeros([20, 13]))
b1 = tf.Variable(tf.zeros([13]))
hidden1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

#this section makes the second hidden layer
W2 = tf.Variable(tf.zeros([13, 8]))
b2 = tf.Variable(tf.zeros([8]))
hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W2) + b2)

#this section makes the output layer
W3 = tf.Variable(tf.zeros([8, 10]))
b3 = tf.Variable(tf.zeros([10]))
y = tf.nn.sigmoid(tf.matmul(hidden2, W3) + b3)

#this sections defines the loss function to minimize 
y_ = tf.placeholder(tf.float32, [None,10])
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(10000):
  print("epoch: "+str(i))
  batch_xs = traindata[0:2**16]#[i*25:25+i*25]
  batch_ys = trainingSums[0:2**																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																											16]#[i*25:25+i*25]
  sess.run(train_step, feed_dict={x: ba																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										tch_xs, y_: batch_ys})
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																														
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: testdata[0:250], y_: testingSums[0:250]}))
