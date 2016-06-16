__author__ = 'billywu'
import tensorflow as tf
import numpy as np
import sys

try:
    # Commandline Input Parameters
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
    training_size=int(cmdargs[11])
    testing_size=int(cmdargs[12])
except:
    # Default Parameters
    learning_rate = 0.001
    training_epochs = 100000
    batch_size = 1000
    display_step = 100
    n_hidden_1 = 1024
    n_hidden_2 = 1024
    n_hidden_3 = 1024
    n_input = 32 # 2 n/2 bit input number
    nbit=n_input/2
    n_classes = 16 # n bit output
    training_size=50000
    testing_size=1000

def primes(n):
    # this method returns all the primes under the integer value n using
    # sieve method without the prime number 2
    ret=[]
    sieve = [True] * (n+1)
    for p in range(3, n+1):
        if (sieve[p]):
            ret.append(p)
            for i in range(p, n+1, p):
                sieve[i] = False
    return ret

def shuffle_in_unison(a, b):
    # this method shuffles the two arrays according to the same randomized
    # index
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

# generates the primes and take 300 random prime numbers primes
prime=np.array(primes(2**nbit))
np.random.shuffle(prime)
prime1=prime[0:300]
prime2=prime[-301:-1]
nop=len(prime)
# sanity check, there should be 300 primes
print "There are", nop, "primes in the range"


def convert(number,bits):
    # this method converts a number into an array containing
    # bit2 values, a bit upper bound must be specified
    ret=[]
    for i in range(0,bits):
        if number%2==0:
            ret.append(-1)
        else:
            ret.append(1)
        number=number/2
    return ret


def min(x,y):
    # find the smaller of the two number
    if x<y:
        return x
    else:
        return y


# The product array contains the n bit semi-primes to be factored,
# and the factor array contains the smaller of the two factors

product=[]
factor=[]
p10=[]
f10=[]
for p1 in prime1:
    for p2 in prime2:
        product.append(convert(p1*p2,nbit*2))
        p10.append(p1*p2)
        factor.append(convert(min(p1,p2),nbit))
        f10.append(min(p1,p2))
product=np.array(product)
factor=np.array(factor)

# randomly select 50,000 semi-primes and their smaller factor to be
# the training data, and a different set of 1000 as testing data

product,factor=shuffle_in_unison(product, factor)
train_x,train_y=product[0:training_size],factor[0:training_size]
test_x,test_y=product[training_size+1:training_size+testing_size],\
              factor[training_size+1:training_size+testing_size]
test10x,test10y=p10[training_size+1:training_size+testing_size],\
                f10[training_size+1:training_size+testing_size]

# save the training and testing partition

np.savetxt("train_x_1.csv",train_x,delimiter=",")
np.savetxt("train_y_1.csv",train_y,delimiter=",")
np.savetxt("test_x_1.csv", test_x, delimiter=",")
np.savetxt("test_y_1.csv", test_y, delimiter=",")

# sanity check: the number of training examples
n_examples=len(train_x)
print "there are", n_examples, "examples in the data"





# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
    layer_3 = tf.nn.sigmoid(layer_3)
    # Output layer with linear activation
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
            batch_x=train_x[i*batch_size:(i+1)*batch_size]
            batch_y=train_y[i*batch_size:(i+1)*batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print "========================================"
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
            print "testing cost", cost.eval({x: test_x, y: test_y})
            result=pred.eval({x: train_x[0:1000], y: train_y[0:1000]})
            counter=0
            correct=0
            for p in result:
                r=[]
                for n in p:
                    if n>0:
                        r.append(1)
                    else:
                        r.append(-1)
                if np.inner(train_y[counter]-np.array(r),train_y[counter]-np.array(r))==0:
                    correct=correct+1
                counter=counter+1
            print "Training correct:", correct
            result=pred.eval({x: test_x, y: test_y})
            counter=0
            correct=0
            for p in result:
                r=[]
                for n in p:
                    if n>0:
                        r.append(1)
                    else:
                        r.append(-1)
                if np.inner(test_y[counter]-np.array(r),test_y[counter]-np.array(r))==0:
                    correct=correct+1
                counter=counter+1
            print "Testing correct:", correct
    print "Optimization Finished!"
    save_path = saver.save(sess, "model_1.ckpt")
    print("Model saved in file: %s" % save_path)