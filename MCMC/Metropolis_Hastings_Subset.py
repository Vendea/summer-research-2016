import tensorflow as tf
import numpy as np
import random
import math
import time

class MCMC:
    def __init__(self, cost, feed, sess, stdev, t0, c, p=1, maximize=False):
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
        self.var_size = reduce(lambda prev,curr: prev + curr.size, self.var, 0)
        self.n = int(self.var_size * p)
        print self.var_size, self.n

    def optimize(self, p=None):
        if p != None:
            self.n = int(self.var_size * p)
        self.t += 1
        var_new = []
        vars_tweaked = sorted(np.random.choice(self.var_size, self.n, replace=False))
        counter = 0
        pos = 0
        for v,t in zip(self.var,self.var_t):
            v_n = []
            for x in np.nditer(v):
                if pos < self.n and counter == vars_tweaked[pos]:
                    v_n.append(random.gauss(x, self.stdev))
                    pos += 1
                else:
                    v_n.append(x)
                counter += 1
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
