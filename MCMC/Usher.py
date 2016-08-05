import tensorflow as tf
import numpy as np
import random
import math

class MCMC:
    def __init__(self, cost, feed, sess, stdev, maximize=False):
        self.cost = cost
        self.feed = feed
        self.var_t = tf.trainable_variables()
        self.var = []
        self.sess = sess
        for t in self.var_t:
            self.var.append(t.eval(session=sess))
        self.prev_cost = sess.run(cost, feed)
        self.stdev = stdev
        self.maximize=maximize

    def optimize(self, stdev=None):
        if stdev != None:
            self.stdev = stdev
        var_new = []
        for v,t in zip(self.var,self.var_t):
            v_n = []
            for x in np.nditer(v):
                v_n.append(random.gauss(x, self.stdev))
            v_n = np.reshape(v_n, v.shape)
            var_new.append(v_n)
            self.sess.run(t.assign(v_n))
        new_cost = self.sess.run(self.cost, self.feed)
        if self.maximize:
            if new_cost > self.prev_cost:
                self.prev_cost = new_cost
                self.var = var_new
            else:
                for v,t in zip(self.var,self.var_t):
                    self.sess.run(t.assign(v))
        else:
            if new_cost < self.prev_cost:
                self.prev_cost = new_cost
                self.var = var_new
            else:
                for v,t in zip(self.var,self.var_t):
                    self.sess.run(t.assign(v))
