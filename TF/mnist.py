# coding: utf-8
import time
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

@pt.Register
def leaky_relu(input_pt):
	return tf.select(tf.greater(input_pt, 0.0), input_pt, 0.01*input_pt)

# 数据
mnist = data_mnist.read_data_sets(one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])
dropout = tf.placeholder(tf.float32, [1])

x_reshape = tf.reshape(x, [-1, 28, 28, 1])
seq = pt.wrap(x_reshape).sequential()

# CNN
# seq.conv2d(6, 16)
# seq.max_pool(2, 2)
# seq.conv2d(6, 16)
# seq.max_pool(2, 2)

def residual(seq, stride, output):
	with seq.subdivide_with(2, tf.add_n) as towers:
		towers[0].conv2d(3, output, stride=stride)
		towers[1].conv2d(3, output, stride=stride).leaky_relu().conv2d(3, output)
	seq.leaky_relu()
	return seq

def residual_with_bt(seq, stride, output):
	with seq.subdivide_with(2, tf.add_n) as towers:
		towers[0].conv2d(3, output, stride=stride).batch_normalize()
		towers[1].conv2d(3, output, stride=stride).batch_normalize().leaky_relu().conv2d(3, output).batch_normalize()
	seq.leaky_relu()
	return seq

# residual
with pt.defaults_scope(activation_fn=None, l2loss=1e-3):
	seq.conv2d(3, 16)
	seq = residual_with_bt(seq, 2, 16)
	seq = residual_with_bt(seq, 1, 16)
	seq = residual_with_bt(seq, 2, 16)
	seq = residual_with_bt(seq, 1, 16)
	seq = residual_with_bt(seq, 2, 32)
	seq.average_pool(4, 4)
	# seq.dropout(dropout[0])
	seq.flatten()
	seq.fully_connected(10)					#TODO: network加上activation_fn=None

softmax, loss = seq.softmax_classifier(10, labels=y)
accuracy = softmax.evaluate_classifier(y)

batch = tf.Variable(0, dtype=tf.float32)
learning_rate = tf.train.exponential_decay(0.05, batch * 64, 64*100, 0.995, staircase=True)
train_op = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(loss, global_step=batch)

# 正确率
def right_rate(sess, data):
	accuracy_arr = []
	for _ in range(int(data.num_examples/64)):
		test_x, test_y = data.next_batch(64)
		acc = sess.run(accuracy, feed_dict={x:test_x, y:test_y, dropout: [1.0]})
		accuracy_arr.append(acc)
	return np.mean(np.array(accuracy_arr))

# GPU使用率
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6    #固定比例
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	for epoch in range(50):
		start_time = time.time()
		lr = 0
		loss_arr = []
		for _ in range(100):
			batch_xs, batch_ys = mnist.train.next_batch(64)
			_, loss_val, lr = sess.run([train_op, loss, learning_rate], feed_dict={x: batch_xs, y: batch_ys, dropout: [0.7]})
			loss_arr.append(loss_val)
		time_cost = time.time() - start_time
		accuracy_value = right_rate(sess, mnist.test)
		loss_mean = np.mean(np.array(loss_arr))
		print("epoch: %2d   time: %5.3f   loss: %5.4f   lr: %5.4f   right: %5.3f%%" %(epoch, time_cost, loss_mean, lr, accuracy_value*100))

# CNN
# time: 2.466   right: 0.365
# time: 1.860   right: 0.605
# time: 1.830   right: 0.745
# time: 1.852   right: 0.86
# time: 1.857   right: 0.885

# Residual
# time: 2.199   right: 0.385
# time: 1.596   right: 0.67
# time: 1.592   right: 0.805
# time: 1.621   right: 0.845
# time: 1.634   right: 0.86
