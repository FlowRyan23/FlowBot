import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import numpy as np
from learning.data_preprocessor import get_feature_sets_ff
from learning.data_preprocessor import get_feature_sets_recurrent

# number of inputs/outputs
n_input = 41
n_output = 14

n_epochs = 1
batch_size = 5000


# simple feedforward (multilayer perceptron) neural net with three hidden layers
def tri_layer_ff_nn():
	# None in shape argument results in 'to be determined'
	# placeholder x as dims: 0.:tbd, 1. n_input=41
	x = tf.placeholder('float', shape=(None, n_input))		 # "shape=" can be omitted; round- and square-brackets interchangable
	y = tf.placeholder('float')

	# amount of nodes in hidden layers
	n_nodes_hl1 = 500
	n_nodes_hl2 = 500
	n_nodes_hl3 = 500

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_input, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_output])),
					'biases': tf.Variable(tf.random_normal([n_output]))}

	# input*weights +biases
	l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)  # relu "rectified linear ?"

	# l1*weights +biases
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	# l2*weights +biases
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return x, y, output


def recurrent_nn():
	x = tf.placeholder('float', [None, n_input, 1])
	y = tf.placeholder('float', [None, n_output])

	n_hidden = 512

	weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_output]))}
	biases = {'out': tf.Variable(tf.random_normal([n_output]))}

	# reduces dimensions of x from 3 to 2
	# -speculation: effect is that values in dim2 are no longer enclosed in list of length 1
	x_ = tf.reshape(x, [-1, n_input])

	# returns n_input=41 tensors
	x_ = tf.split(x_, n_input, 1)

	rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
	outputs, states = rnn.static_rnn(rnn_cell, x_, dtype=tf.float32)

	output = tf.matmul(outputs[-1], weights['out']) + biases['out']

	return x, y, output


# todo runs out of memory, after last epoch is completed, trying to allocate tensor of shape (0.1mil, 512)
def train_neural_network(data_file_name):
	# simple feed forward neural net
	# train_x, train_y, test_x, test_y = get_feature_sets_ff(data_file_name)

	# recurrent neural net
	train_x, train_y, test_x, test_y = get_feature_sets_recurrent(data_file_name)

	print(train_x[0])

	train_set_size = len(train_x)
	test_set_size = len(test_x)
	set_size = train_set_size + test_set_size
	print('Data set size: ', set_size)

	# x, y, output = tri_layer_ff_nn()
	x, y, output = recurrent_nn()
	cost = tf.reduce_sum(tf.abs(tf.subtract(tf.sigmoid(output), tf.sigmoid(y))))

	# Optimizer can have att. learning_rate (defaults to 0.001)
	optimizer = tf.train.AdamOptimizer().minimize(tf.cast(cost, 'float'))

	saver = tf.train.Saver()

	# dynamic vram allocation (supposedly less efficient due to memory fragmentation) has fixed problem: 'CUBLAS_STATUS_ALLOC_FAILED'
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0

			i = 0
			while i < train_set_size:
				start = i
				end = i + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c

				i += batch_size
				print('\tEpoch progress: ', round(end/train_set_size, 2)*100)

			print('Epoch', epoch + 1, 'completed out of', n_epochs, ' loss: ', epoch_loss)

		saver.save(sess, "./Trained_NNs/Flow_Bot_RNN_1M.ckpt")
		print('saved session')

		correct = tf.equal(tf.sigmoid(output), tf.sigmoid(y))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		i = 0
		acc = 0.0
		while i < test_set_size:
			start = i
			end = i + batch_size

			batch_x = np.array(test_x[start:end])
			batch_y = np.array(test_y[start:end])

			acc += accuracy.eval({x: batch_x, y: batch_y})
			i += batch_size

		print('Accuracy: ', acc/(test_set_size/batch_size))


if __name__ == '__main__':
	train_neural_network('all_data')
