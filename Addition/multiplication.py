import random as rand
import math as m
import tensorflow as tf

N = 2
p = 13
int_layer = int(2*N*p-2+m.floor(N/(2**p)-1/(2**p-1)))
out_layer = int(m.log(N*(2**p-1)+1,2))+1

# set up arrays of integers and sums for training and testing
a = [m.floor(rand.random()*2**p) for i in range(0, int(2**p*.4))]
b = [m.floor(rand.random()*2**p) for i in range(0, int(2**p*.4))]
c = [m.floor(rand.random()*2**p) for i in range(0, int(2**p*.6))]
d = [m.floor(rand.random()*2**p) for i in range(0, int(2**p*.6))]
trainSums = [v*u for v in a for u in b]
testSums = [v*u for v in c for u in d]

traindata =   []
testdata =    []
testingSums = []
trainingSums = []
def iconvert(x):
    rval = convert(x,2)
    pad = [0 for i in range(p-len(rval))]
    return pad + rval
def oconvert(x):
    rval = convert(x,2)
    pad = [0 for i in range(p+1-len(rval))]
    return pad + rval
  
def convert(m,n):
    if(m<n):
        return [int(m)]
    else:
        return convert(int(m/n),n) + [int(m%n)]

for i in range(0,250):
    traindata.append(
      iconvert(a[i])+ iconvert(b[i]))
    trainingSums.append( oconvert(trainSums[i]))
    testdata.append(iconvert(c[i]) + iconvert(d[i]))
    testingSums.append(oconvert(testSums[i]))

x = tf.placeholder(tf.float32, [None,N*p])#makes input nodes

#this section makes the first hidden layer
W1 = tf.Variable(tf.zeros([N*p,int_layer]))
b1 = tf.Variable(tf.zeros([int_layer]))
hidden1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

#this section makes the output layer
W2 = tf.Variable(tf.zeros([int_layer, out_layer]))
b2 = tf.Variable(tf.zeros([out_layer]))
y = tf.nn.sigmoid(tf.matmul(hidden1, W2) + b2)

#this sections defines the loss function to minimize 
y_ = tf.placeholder(tf.float32, [None,out_layer])
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(105000):
  batch_xs = traindata
  batch_ys = trainingSums 
                          
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: testdata ,y_: testingSums}))