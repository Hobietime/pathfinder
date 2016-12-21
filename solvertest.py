from MPCsolver import MPCsolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

turns = np.empty((20,1))
turns.fill(.1)
accs = np.empty((20,1))
accs.fill(.00)
targets = np.array([[[ 1, 4, 5, 0, -5, -10]],[[ 1, 9, 6, 10, 9, 10]]])
with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():
		solver = MPCsolver(20)
		global_step = tf.Variable(0, name="global_step", trainable=False)

		optimizer = tf.train.AdamOptimizer(1e-1)
		grads_and_vars = optimizer.compute_gradients(solver.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		sess.run(tf.global_variables_initializer())
		summary_writer = tf.train.SummaryWriter(".", graph=tf.get_default_graph())
		feed_dict = {
			solver.initial_v: 1,
			solver.initial_theta: 0,
			solver.targets: targets,
			#solver.acc: accs
		}
		tf.scalar_summary("loss", solver.loss)
		merged_summary_op = tf.merge_all_summaries()

		_, summary, xpos, ypos, distances = sess.run( [global_step, merged_summary_op, solver.x, solver.y, solver.v], feed_dict)
		summary_writer.add_summary(summary, 1)
		#plt.plot(targets[1,0,:], targets[0,0,:], 'ro',  xerr=1, yerr=1)
		plt.plot(ypos, xpos, 'bo')
		plt.plot(ypos, xpos)
		plt.axis('equal')
		plt.show()
		#print(distances)
		for i in range(100):
			_, itnum, xpos, ypos, acc_loss, velocity_loss, phi_loss, postion_loss, avg_phi, avg_v, loss = sess.run( \
				[train_op, global_step, solver.x, solver.y, solver.acc_loss, solver.velocity_loss, \
				solver.phi_loss, solver.postion_loss, solver.avg_phi, solver.avg_v, solver.loss], feed_dict)
		print("acc_loss, velocity_loss, phi_loss, postion_loss, avg_phi, avg_v, loss")
		print(acc_loss, velocity_loss, phi_loss, postion_loss, avg_phi, avg_v, loss)
		print("done")
		plt.plot(ypos, xpos, 'bo')
		plt.plot(ypos, xpos)
		plt.errorbar(targets[1,0,:], targets[0,0,:], xerr=1.0, yerr=1.0, fmt='o')
		plt.axis('equal')
		plt.show()