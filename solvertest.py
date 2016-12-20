from MPCsolver import MPCsolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

turns = np.empty((20,1))
turns.fill(.1)
accs = np.empty((20,1))
accs.fill(.00)
targets = np.array([[[0, 1, 4, 5]],[[0, 1, 8, 6]]])
with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():
		solver = MPCsolver(20)
		global_step = tf.Variable(0, name="global_step", trainable=False)

		optimizer = tf.train.AdamOptimizer(1e-2)
		grads_and_vars = optimizer.compute_gradients(solver.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		sess.run(tf.global_variables_initializer())
		summary_writer = tf.train.SummaryWriter(".", graph=tf.get_default_graph())
		feed_dict = {
			solver.initial_v: 1,
			solver.initial_theta_v: 0,
			solver.targets: targets,
			#solver.acc: accs
		}
		tf.scalar_summary("loss", solver.loss)
		merged_summary_op = tf.merge_all_summaries()

		_, summary, xpos, ypos, distances = sess.run( [global_step, merged_summary_op, solver.x, solver.y, solver.distances], feed_dict)
		summary_writer.add_summary(summary, 1)
		plt.plot(targets[1,0,:], targets[0,0,:], 'ro')
		plt.plot(ypos, xpos, label = 'Parametric Curve')
		plt.axis('equal')
		plt.show()
		print(distances)
		for i in range(200):
			_, itnum, xpos, ypos, distances = sess.run( [train_op, global_step, solver.x, solver.y, solver.distances], feed_dict)
			print(distances)

		plt.plot(ypos, xpos, label = 'Parametric Curve')
		plt.plot(targets[1,0,:], targets[0,0,:], 'ro')
		plt.axis('equal')
		plt.show()