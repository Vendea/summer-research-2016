__author__ = 'billywu'
__author__ = 'billywu'

from mpi4py import MPI
data_dir='./cifar-10-batches-py'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import os
from sys import path
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]+"/lbfgs"
path.append(p)

import tensorflow as tf
import time
from lbfgs_optimizer import lbfgs_optimizer
from Opserver import Opserver

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 2000
display_step = 1
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

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

with tf.Graph().as_default():
    if rank ==0:
        filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                       for i in xrange(1, 6)]
        for f in filenames:
            if not tf.gfile.Exists(f):
              raise ValueError('Failed to find file: ' + f)
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)
        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_whitening(distorted_image)
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                               min_fraction_of_examples_in_queue)
        # Generate a batch of images and labels by building up a queue of examples.
        images,labels = _generate_image_and_label_batch(float_image, read_input.label,
                                             min_queue_examples, batch_size,
                                             shuffle=True)


    # tf Graph input
    x = tf.placeholder(tf.float32,[batch_size,24,24,3])
    y = tf.placeholder(tf.float32,[batch_size,10])


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
        reshape = tf.reshape(pool2, [batch_size, -1])
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
    init = tf.initialize_all_variables()

    # Launch the graph
    sess=tf.Session()
    sess.run(init)
    bsize=2000/size

    if rank==0:    
        tx = images.eval(session=sess)
        ty = labels.eval(session=sess)
        print ty
        batch_xs = images.eval(session=sess)
        batch_ys = labels.eval(session=sess)
        testx =  images.eval(session=sess)
        testy =  labels.eval(session=sess)
       
        comm.bcast([batch_xs,batch_ys],root=0)
        batch_xs, batch_ys=batch_xs[rank:(rank+1)*bsize], batch_ys[rank:(rank+1)*bsize]
        trainer=lbfgs_optimizer(learning_rate, cost,{x: batch_xs, y: batch_ys},sess,3,comm,size,rank)
        for i in range(100):
            trainer.minimize()
            #print trainer.getFunction(trainer.var)
            print sess.run(accuracy,{x:testx,y:testy}),sess.run(accuracy,{x:tx,y:ty}),sess.run(cost,{x:testx,y:testy}),sess.run(cost,{x:tx,y:ty})
        trainer.kill()
    else:
        batch_xs, batch_ys=comm.bcast([],root=0)
        opServer=Opserver(learning_rate, cost,{x: batch_xs, y: batch_ys},sess,comm,size,rank,0)
        opServer.run()



