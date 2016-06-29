'''
Created on Jun 27, 2016

@author: KatherineMJB
'''
class Incrementer:
    
    import tensorflow as tf
    
    def __init__(self, base):
        self.base = base
        
    def run(self, arr):
        ret = []
        for i in range(0, len(arr), 2):
            carry, place = self.ex(arr[i], arr[i+1]) 
            ret.append(carry)
            ret.append(place)
        return ret
        
    def ex(self, l, r):
        return tf.div(tf.add(l, r), base), tf.mod(tf.add(l,r), base)