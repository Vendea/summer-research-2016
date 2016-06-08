import tensorflow as tf

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

for i in range(1000):
  print("epoch: "+str(i))
  batch_xs = 0
  batch_ys = 0
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))