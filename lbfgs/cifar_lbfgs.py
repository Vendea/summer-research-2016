import tensorflow as tf
from os import listdir
from os.path import isfile, join
import cPickle
import numpy as np
from mpi4py import MPI
import time
from sys import path
from os import getcwd
from lbfgs_optimizer2 import lbfgs_optimizer
from Opserver2 import Opserver


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
NUM_CLASSES = 10


datadir = getcwd()[0:getcwd().rfind("/")]+"/cifar10/cifar-10-batches-py"
train  = [f for f in listdir(datadir) if isfile(join(datadir, f))]
train.pop(train.index("batches.meta"))
test = train.pop(train.index("test_batch"))
train = [datadir+"/"+x for x in train]
test = [datadir+"/"+x for x in [test]]

train = [unpickle(x)for x in train]
test  = [unpickle(x)for x in test]
train_data = []
test_data = []
sess = tf.Session()
total_size = 3
for x in train:
	x.pop("batch_label",None)
 	x.pop("filenames",None)
 	train_data.append(x.pop("data"))
for x in test:
 	x.pop("batch_label",None)
 	x.pop("filenames",None)
 	test_data.append(x.pop("data"))

train_data = reduce(lambda x,y:np.array(list(x)+list(y)),train_data)
train_data = tf.reshape(train_data,[-1])
train_data = tf.transpose(tf.reshape(train_data,[50000, 3,32, 32]),[0,3,2,1])
test_data = tf.reshape(test_data[0],[-1])
test_data = tf.transpose(tf.reshape(test_data,[10000,3, 32, 32]),[0,3,2,1])

reshaped_image = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
height = 24
width = 24
# Image processing for training the network. Note the many random
# distortions applied to the image.

# Randomly crop a [height, width] section of the image.
#distorted_image = tf.random_crop(reshaped_image, [total_size, height, height,3])
# test_data = tf.random_crop(test_data, [total_size, height, height,3])
# # Randomly flip the image horizontally.
# distorted_image = tf.image.random_flip_left_right(distorted_image)

# # Because these operations are not commutative, consider randomizing
# # the order their operation.
# distorted_image = tf.image.random_brightness(distorted_image,
#                                            max_delta=63)
# distorted_image = tf.image.random_contrast(distorted_image,
#                                          lower=0.2, upper=1.8)

# # # Subtract off the mean and divide by the variance of the pixels.
# train_data = tf.image.per_image_whitening(distorted_image)


train_labels = []
for x in range(len(train)):
	temp = []
	for y in train[x]["labels"]:
		temp.append(tf.sparse_to_dense([int(y)],[10],[1]))
	train_labels.append(temp)
train_labels = tf.reshape(train_labels,[-1])
train_labels = tf.reshape(train_labels,[50000,10])
test_labels =[]
for x in range(len(test)):
	temp = []
	for y in test[x]["labels"]:
		temp.append(tf.sparse_to_dense([int(y)],[10],[1]))
	test_labels.append(temp)
test_labels = tf.reshape(test_labels,[-1])
test_labels = tf.reshape(test_labels,[10000,10])

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


# tf Graph input
x = tf.placeholder(tf.float32,[None,32,32,3])
y = tf.placeholder(tf.float32,[None,10])


with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    #conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
# conv2
with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

# local3
with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [-1, 8*8*64])
    #print(reshape.get_shape)
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
# local4
with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

# softmax, i.e. softmax(WX + b)
with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    pred = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess=tf.Session()
sess.run(init)
bsize=400
tx = train_data.eval(session=sess)[0:400]
ty = train_labels.eval(session=sess)[0:400]
testx =  test_data.eval(session=sess)[0:400]
testy =  test_labels.eval(session=sess)[0:400]

if rank==0:
    trainer=lbfgs_optimizer(0.0001, cost,[],sess,1,comm,size,rank)
    for b in range(1):
        data_x=tx[bsize*b:bsize*(b+1)]
        data_y=ty[bsize*b:bsize*(b+1)]
        trainer.update(data_x,data_y,x,y)
        start=time.time()
        for i in range(50):
            c = trainer.minimize()
            if i%2==0:
                train = sess.run(accuracy,{x:tx[0:400],y:ty[0:400]})
                test  = sess.run(accuracy,{x:testx[0:400],y:testy[0:400]})
                trainc=sess.run(cost,{x:tx[0:400],y:ty[0:400]})
                testc= sess.run(cost,{x:testx[0:400],y:testy[0:400]})
                f=trainer.functionEval
                g=trainer.gradientEval
                i=trainer.innerEval
                print i, f, g, train, test, trainc, testc

else:
    opServer=Opserver(0.0001, cost,[],sess,comm,size,rank,0,x,y)
    opServer.run()






