import tensorflow as tf
from DeepNetwork import Convolutional

NUM_CLASSES = 10
x = tf.placeholder(tf.float32, [None, 32,32,3])
conv_layers = 2 
full_layers = 3
def init_w(i,dim=None):
  if dim != None:
    return tf.Variable(tf.random_normal([dim,384]))
  if i == 0:
    return tf.Variable(tf.random_normal([5, 5, 3, 64]))
  elif i == 1:
   return tf.Variable(tf.random_normal([5, 5, 64, 64])) 
  elif i == 4:
    return tf.Variable(tf.random_normal([192, NUM_CLASSES])) 
  else:
     return tf.Variable(tf.random_normal([384, 192])) 

def init_b(i):
  if i < 2:
   return tf.Variable(tf.random_normal([64])) 
  elif  i== 2:
    return tf.Variable(tf.random_normal([384]))   
  elif i == 4:
    return tf.Variable(tf.random_normal([NUM_CLASSES]))
  else: 
    return tf.Variable(tf.random_normal([192])) 
def activation(i):
  if i != conv_layers+full_layers -1:
    return tf.nn.relu
  else:
    return tf.nn.softmax
  
strides_conv = [
  [1, 1, 1, 1],
  [1, 1, 1, 1]]
ksize = [
  [1, 3, 3, 1],
  [1, 3, 3, 1]]
strides_pool = [
[1, 2, 2, 1]
,[1, 2, 2, 1]]


network = Convolutional(x,conv_layers,full_layers,init_w,init_b,activation,strides_conv,ksize,strides_pool)