# coding: utf-8
import time
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

# 数据
mnist = data_mnist.read_data_sets(one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])

x_reshape = tf.reshape(x, [-1, 28, 28, 1])
seq = pt.wrap(x_reshape).sequential()
with pt.defaults_scope(activation_fn=tf.nn.relu):
	with seq.subdivide(2) as towers:
		towers[0].conv2d([7, 7], 16).max_pool(2, 2)
		towers[1].conv2d([6, 6], 16).max_pool(2, 2)
seq.flatten()
seq.fully_connected(32, activation_fn=tf.nn.relu)
seq.fully_connected(10, activation_fn=None)		#TODO: network加上activation_fn=None

softmax, loss = seq.softmax_classifier(10, labels=y)
accuracy = softmax.evaluate_classifier(y)
optimizer = tf.train.GradientDescentOptimizer(0.01)  # learning rate
train_op = pt.apply_optimizer(optimizer, losses=[loss])

# GPU使用率
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6    #固定比例
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	for _ in range(10):
		start_time = time.time()
		for _ in range(10):
			batch_xs, batch_ys = mnist.train.next_batch(32)
			_, loss_val = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys})
		time_cost = time.time() - start_time
		print("time: %5.3f" %(time_cost))

	accuracy_value = sess.run(accuracy, feed_dict={x:mnist.test.images[0:100], y:mnist.test.labels[0:100]})
	print 'Accuracy: %g' % accuracy_value



# time: 0.569
# time: 0.268
# time: 0.252
# time: 0.272
# time: 0.268
# time: 0.266
# time: 0.263
# time: 0.270
# time: 0.270
# time: 0.261
# Accuracy: 0.75

