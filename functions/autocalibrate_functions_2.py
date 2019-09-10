# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import logging
import numpy as np
import tensorflow as tf
# disable eager execution to allow the code, that was built in tensorflow < 2.0, to run with the placeholders.
tf.compat.v1.disable_eager_execution()

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import calculate_moving_average


def find_segments_moving_averages(acc_data, std_threshold, hz, sliding_window = 10):
	"""

	Calculate moving averages of windows of individual acceleration axes

	Parameters
	---------
	acc_data : np.array(samples, axes)
		acceleration data

	Returns
	--------
	segments : np.array
		numpy array with segments that can be candidates for non wear time

	"""

	# adjust the minimum segment length to accomodate the number of samples per second.
	sliding_window *= hz

	# empty array to store the segments
	segments = []

	# loop over the data
	for i in range(0, len(acc_data), sliding_window):

		# segement out the data to be analyzed
		segment = acc_data[i:i + sliding_window]

		# calculate the standard deviation of each acceleration axis
		segment_std = np.std(segment, axis = 0)
		
		# check if the standard deviation of all axes are below the threshold
		if np.all(segment_std <= std_threshold):

			# calculate the moving averages of each axes
			y_mov_avg = np.mean(calculate_moving_average(segment[:,0]))
			x_mov_avg = np.mean(calculate_moving_average(segment[:,1]))
			z_mov_avg = np.mean(calculate_moving_average(segment[:,2]))

			# append to segments list
			segments.append([y_mov_avg, x_mov_avg, z_mov_avg])

	return np.array(segments)


def get_calibration_weights(X_train, learning_rate = 0.0001, training_epochs = 10000, optimizer_type = 'adam'):
	"""
	Get the autocalibration weights by definining a linear transformation function and a cost function

	Parameters
	---------
	X_train : np.array(samples, axes = 3)
		numpy array with acceleration values
	learning_rate : float (optional)
		learning rate for the gradient descent, determines the size of the change of the weights of each update
	training_epochs : int (optional)
		number of passes over all the data samples
	optimizer_type : string (optional)
		either 'adam' or 'none'

	Returns
	--------
	weights : dic()
		dictionary with weights
		
		t_y = theta of Y axis
		t_x = theta of X axis
		t_z = theta of Z axis

		b_y = bias of Y axis
		b_x = bias of X axis
		b_z = bias of Z axis
	"""

	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	# construct Y values, this is basically a list of 1 since the VMU needs to 1 when there is no momvement (1 = gravitational force)
	Y_train = np.ones((len(X_train),1), dtype=float)
	# define the number of samples
	m = len(X_train)


	# tf Graph Input
	X = tf.compat.v1.placeholder('float64', [m, 3])
	Y = tf.compat.v1.placeholder('float64', [m, 1])

	# Set model weights
	W = tf.Variable(np.array([1.,1.,1.]).reshape(1,3), dtype=np.float64)
	b = tf.Variable(np.array([0.,0.,0.]).reshape(1,3), dtype=np.float64)
	
	# Construct a linear model
	# pred = tf.sqrt(tf.reduce_sum(tf.square(tf.add(tf.matmul(X,W),tf.transpose(b))), 1, keepdims=True))
	pred = tf.sqrt(tf.reduce_sum(input_tensor = tf.square(tf.add(tf.multiply(X,W),b)), axis = 1, keepdims = True))
		
	# Mean squared error
	# cost = tf.reduce_sum(tf.pow(pred-Y, 2))/ (2*m)
	cost = 1. / (2 * m) * tf.reduce_sum(input_tensor=tf.pow(pred - Y, 2))

	# set the gradient descent optimizer
	if optimizer_type == 'adam': 
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
	elif optimizer_type == 'none':
		optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	else:
		logging.error('Unknown optimer send as argument: {}. Choose either "adam" or "none"'.format(optimizer_type))
		exit(1)
	
	# Initialize the variables (i.e. assign their default value)
	init = tf.compat.v1.global_variables_initializer()

	# Start training
	with tf.compat.v1.Session() as sess:

		# Run the initializer
		sess.run(init)

		# Fit all training data
		for _ in range(training_epochs):

			# set the feed of data
			feed = { X: X_train, Y: Y_train }

			# run the gradient descent
			sess.run(optimizer, feed_dict=feed)
			# print sess.run(cost, feed_dict=feed) * 1000
		
		# create dictionary of weights
		weights = { 	't_y' : sess.run(W)[0][0],
					 	't_x' : sess.run(W)[0][1],
					 	't_z' : sess.run(W)[0][2],
					 	'b_y' : sess.run(b)[0][0],
					 	'b_x' : sess.run(b)[0][1],
					 	'b_z' : sess.run(b)[0][2]}

		return weights


def calibrate_accelerometer_data(acc_data, weights):
	"""
	calibrate acceleration data

	Parameters
	---------
	acc_data : np.array(samples, 3 = axes)
	weihts : dictionary
		dictionary of weights
	"""

	t_y = weights['t_y']
	t_x = weights['t_x']
	t_z = weights['t_z']
	b_y = weights['b_y']
	b_x = weights['b_x']
	b_z = weights['b_z']

	return acc_data * np.array([t_y, t_x, t_z]) + np.array([b_y, b_x, b_z])


def parse_calibration_weights(dictionary):	
	"""
	Parse the calibration weights from a dictionary

	Parameters
	----------
	dictionary : dict()
		dictionary with key-value pairs (basically meta-data from a dataset within a group)

	Return
	--------
	weights : dict()
		dictionary with t_y, t_x, t_z, b_y, b_x, b_z
	"""

	weights = { 't_y' : dictionary['t_y'],
				't_x' : dictionary['t_x'],
				't_z' : dictionary['t_z'],
				'b_y' : dictionary['b_y'],
				'b_x' : dictionary['b_x'],
				'b_z' : dictionary['b_z']}

	return weights	


def return_default_weights():
	"""
	Return weights that will not change the values of the acceleration data if applied.
	For t_y, t_x, and t_z this is 1, since multiplying with 1 will not change the value
	For b_y, b_x, and b_x this is 0, since adding zero bias will not change the value
	"""

	return { 't_y' : 1., 't_x' : 1., 't_z' : 1., 'b_y' : 0.,'b_x' : 0.,'b_z' : 0.}