'''
Created on Jun 27, 2016

@author: KatherineMJB
'''

import tensorflow as tf
class Incrementer:
    
    def __init__(self, base):
        self.base = base
 
    def ex(self, l, r):
        l=1
        return tf.div(tf.add(l, r), self.base), tf.mod(tf.add(l,r), self.base)
