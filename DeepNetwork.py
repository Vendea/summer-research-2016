import tensorflow as tf 
import numpy as np


class Feed_forward:
	def __init__(self,x,layers,init_w,init_b,activation):
		self._layers          = layers
		self._weights         = [] 
		self._bias            = []
		self._init_w          = init_w
		self._init_b          = init_b
		self._activation      = activation
		for i in range(layers):
			self._weights.append(tf.Variable(self._init_w(i)))
			self._bias.append(tf.Variable(self._init_b(i)))
		self.pred = x
		for w,b in zip(self._weights,self._bias):
			self.pred = self._activation(tf.matmul(self.pred,w)+b)

class Convolutional:
	def __init__(self,x,conv_layers,full_layers,init_w,init_b,activation,strides_conv,ksize,strides_pool):
		self._total_layers = conv_layers + full_layers
		self._conv_layers  = conv_layers
		self._full_layers  = full_layers
		self._weights         = [] 
		self._bias            = []
		self._init_w          = init_w
		self._init_b          = init_b
		self._activation      =  activation

		for i in range(self._conv_layers):
			self._weights.append(tf.Variable(self._init_w(i)))
			self._bias.append(tf.Variable(self._init_b(i)))
		self.pred = x

		for w,b,sc,k,sp,i in zip(self._weights[0:conv_layers],self._bias[0:conv_layers],strides_conv,ksize,strides_pool,range(conv_layers)):
		    conv = tf.nn.conv2d(self.pred, w, sc, padding='SAME')
		    bias = tf.nn.bias_add(conv, b)
		    conv1 = activation(i)(bias)

		    # pool1
		    pool1 = tf.nn.max_pool(conv1, ksize=k, strides=sp,
		                         padding='SAME')
		    # norm1
		    self.pred = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		self.pred = tf.reshape(self.pred, [-1,np.prod(self.pred.get_shape()[1:]).value])

		self._weights.append(tf.Variable(self._init_w(i+1,self.pred.get_shape()[1].value)))
		self._bias.append(tf.Variable(self._init_b(i+1)))
		
		for i in range(conv_layers+1,self._total_layers):
			self._weights.append(tf.Variable(self._init_w(i)))
			self._bias.append(tf.Variable(self._init_b(i)))

		for w,b in zip(self._weights,self._bias):
			print(w.get_shape(),b.get_shape())
		for w,b,i in zip(self._weights[conv_layers:],self._bias[conv_layers:],range(conv_layers,self._total_layers)):
			self.pred = self._activation(i)(tf.matmul(self.pred,w)+b)