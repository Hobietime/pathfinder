import tensorflow as tf
import numpy as np

class MPCsolver(object):
	def __init__(self, num_steps, v_limit=2, a_limit=.01, phi_limit=.2, p_slack=1):
		"""This class can be thought of as the configuration file for the dataflow graph
			takes a set of limits (num_steps, v_limit, a_limit, phi_limit, p_slack) 
			on construction and requires two initial conditions(intial_v, initial_theta, targets)
			it optimizes a series of control inputs (turn, throttle), the length of which is defined
			by num_steps"""
		self.initial_v = tf.placeholder(tf.float32, name="initial_speed")
		self.initial_theta = tf.placeholder(tf.float32, name="initial_theta")
		self.targets = tf.placeholder(tf.float32, [2,1,None], name="targets")

		self.turn = tf.Variable(
				name="turn",
				initial_value=np.zeros(num_steps,  dtype=np.float32).reshape(num_steps,1))
		self.throttle = tf.Variable(
				name="throttle",
				initial_value=np.zeros(num_steps,  dtype=np.float32).reshape(num_steps,1))

		with tf.name_scope("input_translation"):
			self.phi = self.turn
			self.t = self.throttle
			self.v = tf.add(tf.cumsum(self.t), self.initial_v)

		with tf.name_scope("XY_tranlation"):
			self.theta = tf.add(tf.cumsum(self.phi), self.initial_theta)
			self.x_v = tf.multiply(self.v, tf.cos(self.theta))
			self.y_v = tf.multiply(self.v, tf.sin(self.theta))

			self.x = tf.cumsum(self.x_v)
			self.y = tf.cumsum(self.y_v)
			self.xy = tf.pack([self.x,self.y])

		with tf.name_scope("constraints"):
			self.target_distances = tf.reduce_min(tf.reduce_max(tf.abs(tf.subtract(self.xy, self.targets)), axis=0), axis=0, name="postion_constraint")
			self.max_v = tf.reduce_max(tf.abs(self.v))
			self.a = tf.add(tf.multiply(tf.square(tf.abs(self.t)), tf.abs(self.phi)),tf.abs(self.t))
			self.max_a = tf.reduce_max(self.a)
			self.max_phi = tf.reduce_max(tf.abs(self.phi))
			self.avg_phi = tf.reduce_mean(tf.abs(self.phi))
			self.avg_v = tf.reduce_mean(self.v)

		with tf.name_scope("loss"):
			self.acc_loss = tf.nn.relu(tf.subtract(self.max_a, a_limit))
			self.velocity_loss = tf.abs(tf.subtract(self.max_v, v_limit))
			self.phi_loss = tf.nn.relu(tf.subtract(self.max_phi, phi_limit))
			self.postion_loss = tf.reduce_mean(tf.nn.relu(tf.subtract(self.target_distances, p_slack)))

			self.loss = 2*self.postion_loss + 2*self.velocity_loss + 2*self.phi_loss + 2*self.acc_loss - self.avg_v
			
			
