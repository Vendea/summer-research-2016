'''
Created on Jun 27, 2016

@author: KatherineMJB
'''

import tensorflow as tf

class Adder:
    
    def __init__(self, base):
        self.base = base
        
    '''def run(self, tens):
        ret = []
        for i in range(0, len(arr), 2):
            carry, place = self.ex(arr[i], arr[i+1]) 
            ret.append(carry)
            ret.append(place)
        return ret'''
    
    def ex(self, l, r):
        return tf.div(tf.add(l,r),self.base), tf.mod(tf.add(l,r),self.base)
        