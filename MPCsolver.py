import tensorflow as tf

class MPCsolver(object):
	def __init__(self, num_steps):
		self.initial_v = tf.placeholder(tf.float32, name="initial_speed")
		self.initial_theta_v = tf.placeholder(tf.float32, name="initial_theta_v")
		self.targets = tf.placeholder(tf.float32, [2,1,None], name="targets")
		#self.y_v = tf.placeholder(tf.float32, [num_steps,1], name="y_v")
		#self.x_v = tf.placeholder(tf.float32, [num_steps,1], name="theta_v")
		#self.turn = tf.placeholder(tf.float32, [num_steps, 1], name="turn")
		#self.acc = tf.placeholder(tf.float32, [num_steps, 1], name="throttle")
		self.turn = tf.get_variable(
				"turn",
				shape=[num_steps, 1],
				initializer=tf.contrib.layers.xavier_initializer())
		self.throttle = tf.get_variable(
				"throttle",
				shape=[num_steps, 1],
				initializer=tf.contrib.layers.xavier_initializer())

		with tf.name_scope("input_translation":
			self.theta_v = self.turn
			self.v = tf.add(tf.cumsum(self.throttle), self.initial_v)
			self.theta = tf.add(tf.cumsum(self.theta_v), self.initial_theta_v)

		with tf.name_scope("XY_tranlation"):
			self.x_v = tf.multiply(self.v, tf.cos(self.theta))
			self.y_v = tf.multiply(self.v, tf.sin(self.theta))

			self.x = tf.cumsum(self.x_v)
			self.y = tf.cumsum(self.y_v)
			self.xy = tf.pack([self.x,self.y])

		with tf.name_scope("constraints"):
			self.distances = tf.reduce_min(tf.reduce_max(tf.abs(tf.subtract(self.xy, self.targets)), axis=0), axis=0, name="postion_constraint")


		with tf.name_scope("loss"):
			
			self.distance = tf.reduce_mean(self.distances)

			self.losses = tf.reduce_min(self.x)
			self.loss = self.distance
			
			
