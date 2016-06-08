__author__ = 'billywu'
import tensorflow as tf
import numpy as np
import sys

try:
    # Input Parameters
    total=len(sys.argv)
    cmdargs = str(sys.argv)
    learning_rate=float(cmdargs[1])
    training_epochs=int(cmdargs[2])
    batch_size = int(cmdargs[3])
    display_step = int(cmdargs[4])
    n_hidden_1 = int(cmdargs[5])
    n_hidden_2 = int(cmdargs[6])
    n_hidden_3 = int(cmdargs[7])
    n_input = int(cmdargs[8]) # 2 n/2 bit input number
    nbit=int(cmdargs[9])
    n_classes = int(cmdargs[10])
except:
    # Default Parameters
    learning_rate = 0.001
    training_epochs = 10000
    batch_size = 1000
    display_step = 1
    n_hidden_1 = 256
    n_hidden_2 = 256
    n_hidden_3 = 256
    n_input = 32 # 2 n/2 bit input number
    nbit=n_input/2
    n_classes = 16 # n bit output

def primes(n):
    ret=[]
    sieve = [True] * (n+1)
    for p in range(3, n+1):
        if (sieve[p]):
            ret.append(p)
            for i in range(p, n+1, p):
                sieve[i] = False
    return ret

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

prime=np.array(primes(2**nbit))
np.random.shuffle(prime)
prime=prime[0:300]
nop=len(prime)
print "There are", nop, "primes in the range"


def convert(number,bits):
    ret=[]
    for i in range(0,bits):
        ret.append(number%2)
        number=number/2
    return ret


def min(x,y):
    if x<y:
        return x
    else:
        return y

input_data=[]
output_data=[]
for output1 in prime:
    for output2 in prime:
        input_data.append(min(convert(output1,16),convert(output2,16)))
        output_data.append(convert(output1*output2,32))
input_data=np.array(input_data)
output_data=np.array(output_data)
factorization,product=shuffle_in_unison(input_data, output_data)
input_data,output_data=product[0:50000],factorization[0:50000]
test_x,test_y=product[50001:51000],factorization[50001:51000]
n_examples=len(input_data)

print "there are", n_examples, "examples in the data"
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
    layer_3 = tf.nn.relu(layer_3)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = multilayer_perceptron(x, weights, biases)

cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred, y))))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_examples/batch_size)
        for i in range(total_batch):
            batch_x=input_data[i*batch_size:(i+1)*batch_size]
            batch_y=output_data[i*batch_size:(i+1)*batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: test_x, y: test_y})
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)