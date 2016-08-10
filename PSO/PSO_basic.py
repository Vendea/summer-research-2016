from mpi4py import MPI
import tensorflow as tf
import numpy as np
import random
import copy

class PSO:

	def __init__(self, comm, cost, feed, sess, b_up, b_lo, omega, phi_p, phi_g):
		self.comm = comm
		self.cost = cost
		self.feed = feed
		self.var_t = tf.trainable_variables()

		self.pl = [tf.placeholder(x.shape) for x in self.var_t]
		self.single_assign = []
		for t, p in zip(self.var_t, self.pl):
			self.single_assign.append(t.assign(p))

        self.var = []
        self.sess = sess
        for t in self.var_t:
            self.var.append(t.eval(session=sess))
		self.omega = omega
		self.phi_p = phi_p
		self.phi_g = phi_g
		self.num_neighbors = num_neighbors

		self.eval_curr = sess.run(cost, feed)

		self.my_best = copy.deepcopy(var)
		self.eval_my_best = self.eval_curr

		self.g_best = None
		self.eval_g_best = None
		self.update_group_best()

		self.vel = [random.randrange(b_lo-b_up, b_up-b_lo) for _ in np.size(var)]

	def optimize(self):
		pos = 0
		var_new = []
		for sa,v,p,pb,gb in zip(self.single_assign, self.var, self.pl, self.my_best, self.g_best):
			v_n = []
			for x,p,g in zip(np.nditer(v), np.nditer(pb), np.nditer(gb)):
				r_p = random.random()
				r_g = random.random()
				self.vel[pos] = \
					self.omega * self.vel[pos] + \
					self.phi_p * r_p * (p - x) + \
					self.phi_g * r_g * (g - x)
				v_n.append(x + self.vel[pos])
			v_n = np.reshape(v_n, v.shape)
			var_new.append(v_n)
			self.sess.run(sa, {p: v_n})
		self.var = copy.deepcopy(var_new)
		self.eval_curr = self.sess.run(cost, feed)
		if self.eval_curr > self.eval_my_best:									# change for min/max-ing
			self.my_best = copy.deepcopy(var)
			self.eval_my_best = self.eval_curr
		self.update_group_best()

	def update_group_best(self):
		if comm.Get_rank() == 0:
			best_index = np.argmax(self.comm.gather(self.my_best, root=0))		# change for min/max-ing
			self.comm.bcast(best_index, root=0)
		else:
			self.comm.gather(self.my_best, root=0)
			best_index = self.comm.bcast(None, root=0)

		if best_index == comm.Get_rank():
			self.g_best, self.eval_g_best = self.comm.bcast((self.my_best, self.eval_my_best), root=best_index)
		else:
			self.g_best, self.eval_g_best = self.comm.bcast(None, root=best_index)
