import tensorflow as tf
import numpy as np
import random
import math

class MCMC:
    def __init__(self, cost, feed, sess, stdev, t0, c, maximize=False):
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
        self.t0 = t0
        self.t = t0 - 1
        self.c = c

    def optimize(self, stdev=None):
        if stdev != None:
            self.stdev = stdev
        self.t += 1
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
            if min(1, self.scale_cost(new_cost)/self.scale_cost(self.prev_cost)) > random.random():
                self.prev_cost = new_cost
                self.var = var_new
            else:
                for v,t in zip(self.var,self.var_t):
                    self.sess.run(t.assign(v))
        else:
            if min(1, self.scale_cost(self.prev_cost)/self.scale_cost(new_cost)) > random.random():
                self.prev_cost = new_cost
                self.var = var_new
            else:
                for v,t in zip(self.var,self.var_t):
                    self.sess.run(t.assign(v))

    def scale_cost(self, x):
        return math.exp(self.c * x * math.log(self.t))
