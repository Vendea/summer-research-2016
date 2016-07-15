	def HMC(U,epsilon,L,sess,feed):
		current_q = tf.trainable_variables()
		q         = current_q
		p         = np.random.normal(0, 1, length(q))
		current_p = p 
		p 		  = p - epsilon * tf.Variable(tf.gradients(q))/2
		for i in range(L):
			q = q + epsilon * p
			if i != L:
				p = p - epsilon * tf.Variable(tf.gradients(q))
		p = p - epsilon * tf.Variable(tf.gradients(q))/2
		p = -p 
		current_U = U(current_q)
		current_K = np.sum(current_p**2)/2
		proposed_U = U(q)
		proposed_K = np.sum(p**2)/2

		s = np.random.normal(0, 1, 1)
		if s < np.exp(current_U - proposed_U +current_K - proposed_K):
			return q
		else:
			return current_q