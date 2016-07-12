import tensorflow as tf
import numpy as np
import random
from mpi4py import MPI

class MCMC:
    def __init__(self, cost, feed, sess, root, comm):
        self.cost = cost
        self.feed = feed
        self.var_t = tf.trainable_variables()
        self.var = []
        self.sess = sess
        for t in self.var_t:
            self.var.append(t.eval(session=sess))
        self.prev_cost = sess.run(cost, feed)
        self.comm = comm
        self.root = root

    def optimize(self, stdev):
        if self.comm.Get_rank() == self.root:
            self.master_optimize(stdev)
        else:
            self.slave_optimize(stdev)

    def master_optimize(self, stdev):
        (y_j, pi) = self.new_state(self.var, stdev)
        w_picks = self.comm.gather(pi, root=self.root) # w(x, y) = pi(x)

        y_w_sum = sum(w_picks)
        pick = self.rand_pick(w_picks, y_w_sum)
        self.comm.bcast(pick, root=self.root)
        y = self.comm.bcast(y_j, root=pick)

        x_w_sum = self.comm.reduce(self.prev_cost, op=MPI.SUM, root=self.root)
        
        accept = random.random() < min(1, y_w_sum/x_w_sum)
        self.comm.bcast((accept, w_picks[pick]), root=self.root)
        if accept:
            self.accept(y, w_picks[pick])
        else:
            self.reject()

    def accept(self, y, w):
        self.prev_cost = w
        self.var = y

    def reject(self):
        for v,t in zip(self.var,self.var_t):
            self.sess.run(t.assign(v))

    def rand_pick(self, w_picks, w_sum):
        pos = 0
        pos_pick = random.random() * w_sum
        pick = 0
        for i in range(len(w_picks)):
            pos += w_picks[i]
            if pos_pick < pos:
                pick = i
                break
        return pick

    def slave_optimize(self, stdev):
        (y_j, y_pi) = self.new_state(self.var, stdev)
        self.comm.gather(y_pi, root=self.root)

        pick = self.comm.bcast(None, root=self.root)
        y = self.comm.bcast(y_j, root=pick)

        (x_j, x_pi) = self.new_state(y, stdev)
        self.comm.reduce(x_pi, op=MPI.SUM, root=self.root)

        (accept, y_w) = self.comm.bcast(None, root=self.root)
        if accept:
            self.accept(y, y_w)
        else:
            self.reject()

    def new_state(self, var_k, stdev):
        var_new = []
        for v,t in zip(var_k, self.var_t):
            v_n = []
            for x in np.nditer(v):
                v_n.append(random.gauss(x, stdev))
            v_n = np.reshape(v_n, v.shape)
            var_new.append(v_n)
            self.sess.run(t.assign(v_n))
        new_cost = self.sess.run(self.cost, self.feed)
        return (var_new, new_cost)
