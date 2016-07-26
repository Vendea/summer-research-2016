from svhn import read_data_sets
import tensorflow as tf
from mpi4py import MPI
#Parameters
NUM_CLASSES = 10  
learning_rate = .001
num_epochs = 100
#End Parameters
from sys import path
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]+"/lbfgs"
path.append(p)

p = getcwd()[0:getcwd().rfind("/")]+"/Logger"
path.append(p)
import Logger
logfile = Logger.DataLogger("SVHN_LBFGS","Epoch,time,train_accuaracy,test_accuaracy,train_cost,test_cost")
import timer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
svhn = read_data_sets("/tmp/data")

config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 0},
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
sess=tf.Session(config=config)
sess.run(init)
tx,ty = svhn.train.images,svhn.train.labels
train_size =  len(tx)
bsize=train_size
start = time.time()
if rank==0:
    trainer=lbfgs_optimizer(0.0001, cost,[],sess,1,comm,size,rank)
    for b in range(5):
        data_x=tx[bsize*b:bsize*(b+1)]
        data_y=ty[bsize*b:bsize*(b+1)]
        trainer.update(data_x,data_y,x,y)
        start=time.time()
        for i in range(40):
            c = trainer.minimize()
            train=sess.run(accuracy,{x:tx,y:ty})
            test= sess.run(accuracy,{x:svhn.test.images,y:svhn.test.labels})
            train_cost=c
            test_cost= sess.run(cost,{x:svhn.test.images,y:svhn.test.labels})
            #f=trainer.functionEval
            #g=trainer.gradientEval
            #i=trainer.innerEval
            #print i, f, g, train, test,train_cost,test_cost
            logfile.writeData((i,time.time()-start, train, test,train_cost,test_cost))
else:
    opServer=Opserver(0.0001, cost,[],sess,comm,size,rank,0,x,y)
    opServer.run()

