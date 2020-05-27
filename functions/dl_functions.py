# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import logging
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow.keras.callbacks import EarlyStopping # stop training early if it does not further improve
from functions.helper_functions import save_pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config = config)

from functions.helper_functions import create_directory

def standard_scaler(X, copy = True, with_mean = True, with_std = True):
	"""
	Standardize features by removing the mean and scaling to unit variance

	The standard score of a sample x is calculated as:

	z = (x - u) / s
	where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

	Parameters
	----------
	X : numpy array
		array with features
	copy : boolean, optional, default True
		If False, try to avoid a copy and do inplace scaling instead. This is not guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.
	with_mean : boolean, True by default
		If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.
	with_std : boolean, True by default
		If True, scale the data to unit variance (or equivalently, unit standard deviation).

	Returns
	----------
	X : np.array
		scaled numpy array
	"""

	# define the scaler with options
	scaler = StandardScaler(copy, with_mean, with_std)

	# fit the data
	scaler.fit(X)
	
	# scale the data
	return scaler.transform(X)


def polynomial_features(X, degree = 2):
	"""
	Calculate polynomial features of X

	Parameters
	----------
	X : np.array
		numpy array with features
	degree : int (optional)
		polynomial degree to create features for. Defaults to 2

	Returns
	----------
	X : np.array
		numpy array with additional polynomial features
	"""

	poly = PolynomialFeatures(degree)

	return poly.fit_transform(X)


def train_mlp_classifier(X, Y):
	"""
	Train a Mulitlayer Perceptron

	Parameters
	----------
	X : np.array
		numpy array with features
	Y : np.array
		numpy array with class labels or target values

	"""

	# define settings of training
	buffer_size = 1024
	batch_size = 512
	epoch = 10

	# define training and development split percentage
	train_split, dev_split = 0.7, 0.2
	logging.info('Training split : {}, development split : {}'.format(train_split, dev_split))
	train_size, dev_size = int(len(X) * train_split), int(len(X) * dev_split)
	logging.info('Training size : {}, development size : {}'.format(train_size, dev_size))

	# create train, dev, test set
	X_train, X_dev, X_test = X[:train_size], X[train_size:train_size + dev_size], X[train_size + dev_size:] 
	Y_train, Y_dev, Y_test = Y[:train_size], Y[train_size:train_size + dev_size], Y[train_size + dev_size:]

	# trim down X_train to have equal size batches
	batch_trim = len(X_train) % batch_size
	X_train = X_train[:-batch_trim]
	Y_train = Y_train[:-batch_trim]

	# create tensorflow training dataset
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
	# shuffle and create batches
	train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

	# create tensorflow development dataset
	dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev))
	# shuffle and create batches
	dev_dataset = dev_dataset.shuffle(buffer_size).batch(batch_size)


	# create tensorflow development dataset
	test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
	# shuffle and create batches
	test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)

	# use multi GPUs
	mirrored_strategy = tf.distribute.MirroredStrategy()

	# context manager for multi-gpu
	with mirrored_strategy.scope():

		# create sequential model
		model = keras.models.Sequential()

		# flatten the input
		model.add(keras.layers.Flatten(input_shape = (1,7)))

		# dense layer
		model.add(keras.layers.Dense(20, activation = 'relu'))

		# dense layer
		model.add(keras.layers.Dense(20, activation = 'relu'))

		# dense layer
		model.add(keras.layers.Dense(20, activation = 'relu'))

		# dense layer
		model.add(keras.layers.Dense(20, activation = 'relu'))

		# dense layer
		model.add(keras.layers.Dense(20, activation = 'relu'))

		# final layer
		model.add(keras.layers.Dense(1, activation = 'sigmoid'))

		# compile the model
		model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

		# fit the model
		model.fit(train_dataset, epochs = epoch, validation_data = dev_dataset)

		# evaluate on test set
		model.evaluate(test_dataset)

		"""
		TODO
		- save history
		- save model
		- perform grid search on layers and number of neurons
		"""

def create_1d_cnn_non_wear_episodes(X, Y, save_model_folder, model_name, cnn_type, epoch, train_split, dev_split, return_model = False):

	# define settings of training
	buffer_size = 64
	batch_size = 32

	# define training and development split percentage
	logging.info('Training split : {}, development split : {}'.format(train_split, dev_split))
	train_size, dev_size = int(len(X) * train_split), int(len(X) * dev_split)
	logging.info('Training size : {}, development size : {}'.format(train_size, dev_size))

	# create train, dev, test set
	X_train, X_dev, X_test = X[:train_size], X[train_size:train_size + dev_size], X[train_size + dev_size:] 
	Y_train, Y_dev, Y_test = Y[:train_size], Y[train_size:train_size + dev_size], Y[train_size + dev_size:]

	# trim down X_train to have equal size batches
	batch_trim = len(X_train) % batch_size
	if batch_trim != 0:
		X_train = X_train[:-batch_trim]
		Y_train = Y_train[:-batch_trim]

	# trim down X_dev to have equal sized batches
	batch_trim = len(X_dev) % batch_size
	if batch_trim != 0:
		X_dev = X_dev[:-batch_trim]
		Y_dev = Y_dev[:-batch_trim]

	# trim down X_test to have equal sized batches
	batch_trim = len(X_test) % batch_size
	if batch_trim != 0:
		X_test = X_test[:-batch_trim]
		Y_test = Y_test[:-batch_trim]

	logging.info(f'X_train : {X_train.shape}, Y_train: {Y_train.shape}')
	logging.info(f'X_dev : {X_dev.shape}, Y_dev: {Y_dev.shape}')
	logging.info(f'X_test : {X_test.shape}, Y_test: {Y_test.shape}')

	# create tensorflow training dataset
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
	# shuffle and create batches
	train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

	# create tensorflow development dataset
	dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev))
	# shuffle and create batches
	dev_dataset = dev_dataset.shuffle(buffer_size).batch(batch_size)

	# create tensorflow development dataset
	test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
	# shuffle and create batches
	test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)

	# use multi GPUs
	mirrored_strategy = tf.distribute.MirroredStrategy()

	METRICS = [
		keras.metrics.TruePositives(name='tp'),
		keras.metrics.FalsePositives(name='fp'),
		keras.metrics.TrueNegatives(name='tn'),
		keras.metrics.FalseNegatives(name='fn'), 
		keras.metrics.BinaryAccuracy(name='accuracy'),
		keras.metrics.Precision(name='precision'),
		keras.metrics.Recall(name='recall'),
		keras.metrics.AUC(name='auc'),
	]


	# empty list to add callbacks to
	callback_list = []
	# early stopping callback
	callback_list.append(EarlyStopping(monitor = 'val_loss', restore_best_weights = True, min_delta = 0, patience = 25, verbose = 1, mode = 'auto'))

	# context manager for multi-gpu
	with mirrored_strategy.scope():

		# create sequential model
		model = keras.models.Sequential()

		if cnn_type == 'v1':
			
			model.add(keras.layers.Conv1D(filters = 10, kernel_size = 10, activation='relu', input_shape = X.shape[1:]))
			model.add(keras.layers.Flatten())
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(1, activation = 'sigmoid'))

		elif cnn_type == 'v2':

			model.add(keras.layers.Conv1D(filters = 10, kernel_size = 10, activation='relu', input_shape = X.shape[1:]))
			model.add(keras.layers.Conv1D(filters = 10, kernel_size = 10, activation='relu'))
			model.add(keras.layers.Conv1D(filters = 10, kernel_size = 10, activation='relu'))
			model.add(keras.layers.Flatten())
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(1, activation = 'sigmoid'))

		elif cnn_type == 'v3':

			model.add(keras.layers.Conv1D(filters = 20, kernel_size = 50, activation='relu', input_shape = X.shape[1:]))
			model.add(keras.layers.Conv1D(filters = 20, kernel_size = 50, activation='relu'))
			model.add(keras.layers.Conv1D(filters = 20, kernel_size = 50, activation='relu'))
			model.add(keras.layers.Flatten())
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(1, activation = 'sigmoid'))

		elif cnn_type == 'v4':
			
			model.add(keras.layers.Conv1D(filters = 10, kernel_size = 10, activation='relu', input_shape = X.shape[1:]))
			model.add(keras.layers.MaxPooling1D(pool_size = 2))
			model.add(keras.layers.Conv1D(filters = 10, kernel_size = 10, activation='relu'))
			model.add(keras.layers.MaxPooling1D(pool_size = 2))
			model.add(keras.layers.Conv1D(filters = 10, kernel_size = 10, activation='relu'))
			model.add(keras.layers.MaxPooling1D(pool_size = 2))
			model.add(keras.layers.Flatten())
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(50, activation='relu'))
			model.add(keras.layers.Dense(1, activation = 'sigmoid'))


		# compile the model
		model.compile(optimizer = keras.optimizers.Adam(lr = 1e-3), loss = keras.losses.BinaryCrossentropy(), metrics = METRICS)

		# fit the model
		history = model.fit(train_dataset, epochs = epoch, validation_data = dev_dataset, callbacks = callback_list)#, class_weight = class_weight)

		# evaluate on test set
		history_test = model.evaluate(test_dataset)
		# create dataframe
		df_history_test = pd.DataFrame(history_test, index = ['test_loss', 'test_tp', 'test_fp', 'test_tn', 'test_fn', 'test_accuracy', 'test_precision', 'test_recall', 'test_auc'] )

		# create save folder if not exists
		create_directory(os.path.join(save_model_folder, cnn_type))

		# save the model
		model.save(os.path.join(save_model_folder, cnn_type, model_name))

		pd.DataFrame(history.history).to_csv(os.path.join(save_model_folder, cnn_type, f'{model_name}_history.csv'))

		# save test history
		df_history_test.to_csv(os.path.join(save_model_folder, cnn_type, f'{model_name}_history_test.csv'))
		
		# return model if return_model set to True
		if return_model:
			return model, history

def load_tf_model(model_location, show_model_output = False):

	# Recreate the exact same model, including its weights and the optimizer
	new_model = tf.keras.models.load_model(model_location)

	# Show the model architecture
	if show_model_output:
		new_model.summary()

	return new_model


