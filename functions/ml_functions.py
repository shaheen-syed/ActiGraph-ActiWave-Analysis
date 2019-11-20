# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import pickle
import glob2
import os
import random
import csv
import logging
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
	

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import save_pickle, save_dic_to_csv


def create_train_test_split(X, Y, test_size = .2, shuffle = True, random_state = 42, **kwargs):
	"""
	Create a training and test split of the data
	https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

	Parameters
	----------
	X : np.array()
		numpy array with data features
	Y : np.array()
		numpy array with labels
	test_size : float
		percentage of the test size in relation to the total number of training samples
	shuffle : Boolean (optional)
		shuffle the data before splitting into train and test
	random_state : Boolean (optional)
		seed for randomness, can be used to replicate the same split of the data
	"""

	# split data into train and test
	return train_test_split(X, Y, test_size = test_size, shuffle = shuffle, random_state = random_state, **kwargs)

def return_k_folds(n_splits, shuffle = True, random_state = 42):
	"""
	K-Folds cross-validator
	https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

	Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).

	Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
	"""

	return KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)


def return_stratified_k_folds(n_splits, shuffle = False, random_state = 42):
	"""
	Stratified K-Folds cross-validator
	https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

	Provides train/test indices to split data in train/test sets.

	This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
	"""
	return StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
	

def calculate_precision_recall_f1_support(y, y_hat, **kwargs):
	"""
	Compute precision, recall, F-measure and support for each class. Also see: 
	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support

	Paramaters
	---------
	y : np.array(n_samples, 1)
		true class labels
	y_hat : np.array(n_samples, 1)
		predicted class labels
	**kwargs: see scikit learn documentation

	Returns
	---------
	precision : float 
		(if average is not None) or array of float, shape = [n_unique_labels]
	recall : float 
		(if average is not None) or array of float, , shape = [n_unique_labels]
	fbeta_score : float 
	(if average is not None) or array of float, shape = [n_unique_labels]
	support : int 
		(if average is not None) or array of int, shape = [n_unique_labels]. The number of occurrences of each label in y_true.
	"""
	
	return sklearn.metrics.precision_recall_fscore_support(y, y_hat, **kwargs)


def calculate_roc_curve(y, y_hat, **kwargs):
	"""
	Compute Receiver operating characteristic (ROC). See also:
	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

	Paramaters
	---------
	y : np.array(n_samples, 1)
		true class labels
	y_hat : np.array(n_samples, 1)
		predicted class labels
	**kwargs: see scikit learn documentation

	Returns
	--------
	fpr : array, shape = [>2]
		Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
	tpr : array, shape = [>2]
		Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
	thresholds : array, shape = [n_thresholds]
		Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.
	"""

	return sklearn.metrics.roc_curve(y, y_hat, **kwargs)


def calculate_roc_auc(y, y_hat, **kwargs):
	"""
	Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
	Note: this implementation is restricted to the binary classification task or multilabel classification task in label indicator format.See also:
	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
	
	Paramaters
	---------
	y : np.array(n_samples, 1)
		true class labels
	y_hat : np.array(n_samples, 1)
		predicted class labels
	**kwargs: see scikit learn documentation

	Returns
	---------
	auc : float
		the calculated area under the curve
	"""

	return sklearn.metrics.roc_auc_score(y, y_hat, **kwargs)


def get_confusion_matrix(y, y_hat, **kwargs):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification. See also:
	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html


	Paramaters
	---------
	y : np.array(n_samples, 1)
		true class labels
	y_hat : np.array(n_samples, 1)
		predicted class labels
	**kwargs: see scikit learn documentation

	Returns
	---------
	confusion_matrix : np.array(n_classes, n_classers). 
		Confusion matrix
	"""

	return sklearn.metrics.confusion_matrix(y, y_hat, **kwargs)


def get_gridsearch_number_between(random, start, end, length = 1):
	"""
	Get an array of floats between two values (start and stop) random or a structured range

	Parameters
	----------
	random : Boolean
		return random values or range
	start: int/float
		the start of the range
	end: int/float
		the end of the range
	length: int (optional)
		the number of values to generate

	Returns
	--------
	random_grid_search_values : np.array()
		numpy array of floats between start and end of length=length
	"""

	if random:
		return [np.random.uniform(start, end) for x in range(0,length)]
	else:
		return np.linspace(start, end, length)


def create_parameter_grid(grid_setup):
	"""
	Creates the parameter grid dictionary to will be passed into the scikit learn gridsearchcv method
	
	Example:
		grid_setup = {	'LinearSVC' : [{'hyperparameter' : 'C', 
							'random' : False, 
							'min' : -5, 
							'max' : 10, 
							'length' : 100,
							'scaler' : True
							'scaler_order' : 1}],
			
			'SVC' : 	[	{'hyperparameter' : 'C', 
							'random' : False, 
							'min' : -5, 
							'max' : 10, 
							'length' : 100 },

							{'hyperparameter' : 'gamma', 
							'random' : True,
							'min' : -5, 
							'max' : 10, 
							'length' : 100 },

							{'hyperparameter' : 'sublinear_ngram_range',
							'min' : 1,
							'max' : 4}
						] 
			}
	"""

	# empty parameter grid dictionary
	param_grid = {}

	# loop over the rows
	for row in grid_setup:

		if row['hyperparameter'] == 'C':
			param_grid.update({'classify__C': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
		
		if row['hyperparameter'] == 'gamma':
			param_grid.update({'classify__gamma': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
		
		if row['hyperparameter'] == 'alpha':
			param_grid.update({'classify__alpha': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
		
		if row['hyperparameter'] == 'fit_prior':
			param_grid.update({'classify__fit_prior' : [True, False]})

		if row['hyperparameter'] == 'dual':
			param_grid.update({'classify__dual' : [True, False]})

		if row['hyperparameter'] == 'loss':
			param_grid.update({'classify__loss' : ['hinge', 'squared_hinge']})

		if row['hyperparameter'] == 'fit_intercept':
			param_grid.update({'classify__fit_intercept' : [True, False]})

		if row['hyperparameter'] == 'fit_shrinking':
			param_grid.update({'classify__shrinking' : [True, False]})

		if row['hyperparameter'] == 'kernel':
			# param_grid.update({'classify__kernel' : ['linear', 'poly', 'rbf', 'sigmoid']})
			param_grid.update({'classify__kernel' : ['linear']})


		"""
			DESCISION TREES
		"""

		if row['hyperparameter'] == 'criterion':
			param_grid.update({'classify__criterion' : ['gini','entropy']})

		if row['hyperparameter'] == 'splitter':
			param_grid.update({'classify__splitter' : ['best','random']})

		if row['hyperparameter'] == 'max_features':
			param_grid.update({'classify__max_features' : ['auto','sqrt', 'log2', None]})

		if row['hyperparameter'] == 'n_estimators':
			param_grid.update({'classify__n_estimators' : [int(x) for x in get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length'])]})

		if row['hyperparameter'] == 'algorithm':
			param_grid.update({'classify__algorithm' : ['SAMME','SAMME.R']})

		if row['hyperparameter'] == 'adaboost_learning_rate':
			param_grid.update({'classify__learning_rate' : range(row['min'], row['max'])})			


		"""
			VECTORIZER
		"""
		if row['hyperparameter'] == 'sublinear_tf':
			param_grid.update({'vectorizer__sublinear_tf': [True, False]})
		
		if row['hyperparameter'] == 'sublinear_use_idf':
			param_grid.update({'vectorizer__use_idf': [True, False]})
		
		if row['hyperparameter'] == 'sublinear_ngram_range':
			param_grid.update({'vectorizer__ngram_range': zip(np.repeat(row['min'], row['max'] + 1), range(row['min'], row['max'] + 1))})

		if row['hyperparameter'] == 'min_df':
			param_grid.update({'vectorizer__min_df': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
	
		if row['hyperparameter'] == 'max_df':
			param_grid.update({'vectorizer__max_df': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })

			
	return param_grid


def get_classifier(classifier, random_state = None):
	"""
	Return the appropriate classifier method that scikit learn uses

	Parameters
	-----------
	classifier : string
		the name of the classifier, basically the name of the method in string

	Returns:
	classifier : scikit learn method
	"""

	if classifier == 'SVC':
		return svm.SVC(probability = True, max_iter = 10000000, random_state = random_state) # class_weight = 'balanced'
	elif classifier == 'LinearSVC':
		return svm.LinearSVC(max_iter = 1000000, random_state = random_state)
	elif classifier == 'LogisticRegression':
		return LogisticRegression(max_iter = 1000000, random_state = random_state)
	elif classifier == 'MultinomialNB':
		return MultinomialNB()
	elif classifier == 'BernoulliNB':
		return BernoulliNB()
	elif classifier == 'DecisionTreeClassifier':
		return DecisionTreeClassifier(random_state = random_state)
	elif classifier == 'AdaBoostClassifier':
		return AdaBoostClassifier(random_state = random_state)
	else:
		logging.error('Classifier {} not part of classifier list'.format(classifier))
		exit(1)


def create_pipeline(classifier, pipeline_setup):
	"""
	Dynamically create the pipeline

	Parameters
	----------
	classifier : string
		name of the classifier algorithm to use
	pipeline_setup : dictionary
		additional options for the pipeline, such as for instance a vectorizer

	Returns
	--------
	pipeline : sklearn pipeline
		steps to execute within a machine learning pipeline
	"""

	# create pipeline with classifier
	pipeline = Pipeline([('classify', get_classifier(classifier))])

	# add processes to the pipeline
	for row in pipeline_setup:

		if row.get('scaler'):
			
			"""
				Scikit learn standardscaler : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

				copy : boolean, optional, default True
				If False, try to avoid a copy and do inplace scaling instead. This is not guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.

				with_mean : boolean, True by default
				If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.

				with_std : boolean, True by default
				If True, scale the data to unit variance (or equivalently, unit standard deviation).
			"""
			pipeline.steps.insert(row['scaler_order'],['scaler', StandardScaler()])

		if row.get('poly_features'):

			"""
				Scikit learn polynomialFeatures: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
				
				degree : integer
				The degree of the polynomial features. Default = 2.

				interaction_only : boolean, default = False
				If true, only interaction features are produced: features that are products of at most degree distinct input features (so not x[1] ** 2, x[0] * x[2] ** 3, etc.).

				include_bias : boolean
				If True (default), then include a bias column, the feature in which all polynomial powers are zero (i.e. a column of ones - acts as an intercept term in a linear model).

				order : str in {‘C’, ‘F’}, default ‘C’
				Order of output array in the dense case. ‘F’ order is faster to compute, but may slow down subsequent estimators.

			"""

			pipeline.steps.insert(row['poly_features_order'],['poly_features', PolynomialFeatures(degree = row['poly_features_degree'], include_bias = False)])

	return pipeline


def execute_gridsearch_cv(X, Y, test_size, shuffle, pipeline_setup, grid_setup, cv, n_jobs, scoring, verbose, save_model, model_save_location):
	"""
	Execute a grid search cross validated classification model creation

	Parameters
	----------
	X : np.array((n_samples, features))
		array with feature values
	Y : np.array((n_samples, 1))
		class label assignment
	test_size: float
		percentage of the test dataset (for instance 0.2 for 20% of the data for testing)
	shuffle : Boolean
		set to True if the training/test dataset creation need shuffling first
	pipeline_setup: dictionary
		settings for the pipeline, for instance, use scaling or not?
	grid_setup: dictionary
		settings for the grid search, for instance, the values for hyperparameters etc.
	cv: int
		number of fold within cross validation
	n_jobs: int
		number of parallel threads to create model
	scoring: string
		scoring function for cross validation, for isntance, f1_weighted
	verbose: int
		debug output, 10 is most
	save_model : Boolean
		if model needs to be saved to disk
	model_save_location: os.path
		if model needs to be saved, this is the location.
	"""

	try:

		# random_state = 2740 gives 41 samples of 0 in test
		# random state 396 gives 42 samples in test that are non-wear time

		# testing of randomness
		top_i, top_n = 0, 0
		for i in range(10000):

			X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = i, shuffle = shuffle)
	
			# number of zeros in y-test
			value = (y_test == 0).sum()

			if value > top_n:
				top_n = value
				top_i = i

		logging.info('Setting random state to {} to have {} zeros in test'.format(top_i, top_n))

		# top_i = 4682
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = top_i, shuffle = shuffle)
		
		# execute gridsearch for each classifier
		for classifier in grid_setup.keys():

			# get parameter grid values
			param_grid = create_parameter_grid(grid_setup[classifier])

			# create pipeline
			pipeline = create_pipeline(classifier, pipeline_setup[classifier])

			# perform K-fold Cross validation
			grid_search = GridSearchCV(pipeline, cv = cv, n_jobs = n_jobs, param_grid = param_grid, scoring = scoring, verbose = verbose)

			# fit the grid_search model to the training data
			grid_search.fit(X_train,y_train)

			# calculate scores on test set
			precision, recall, f1, _ = calculate_precision_recall_f1_support(y_test, grid_search.predict(X_test), average='weighted')
			
			# get the confusion matrix
			cf_matrix = confusion_matrix(y_test, grid_search.predict(X_test))

			# create the results
			results = {'training_f1' : grid_search.best_score_,
						'best_parameters' : grid_search.best_params_,
						'test_precision' : precision,
						'test_recall' : recall,
						'test_f1' : f1,
						'test_confusion_matrix' : cf_matrix}

			# save_model(model = grid_search, file_name = classifier, folder = model_save_location)
			if save_model:
				# save a pickle
				save_pickle(obj = grid_search, file_name = classifier, folder = model_save_location)
				# save results as csv
				save_dic_to_csv(results, file_name = classifier, folder = model_save_location)
	except Exception as e:
		logging.error('Error executing gridsearch CV: {}'.format(e))
		return


def predict_class(classifier, X):
	"""
	Predict class given features

	Parameters
	----------
	classifier : scikit learn classifier
		trained ML model
	X : np.array()
		array with X-features

	Returns
	-------
	classification : int
		inferred class label
	"""

	return classifier.predict(X)


def calculate_classification_performance(tn, fp, fn, tp):
	"""
	Calculate classification performance measures such as precision, recall, f1 from a binary confusion matrix output

	Parameters
	----------
	tn : int
		true negatives
	fp : int
		false positives
	fn : int
		false negatives
	tp : int
		true positives

	Returns
	---------
	classification_performance : dict()
		dictionary with precision, specificity, recall, accuracy, f1, ppv, npv
	"""

	# calculate precision
	precision = tp / (tp + fp)
	# calculate specificity
	specificity = tn / (fp + tn)
	# calculate recall
	recall = tp / (tp + fn)
	# calculate accuracy
	accuracy = (tp + tn) / (tp + tn + fp + fn) 
	# calculate f1
	f1 = 2 * tp / (2 * tp + fp + fn)
	# calculate ppv positive predictive value
	ppv = tp / (tp + fp)
	# calculate npv negative predictive value
	npv = tn / (tn + fn)

	# create dictionary for meta data
	classification_performance = {	'tn' : tn,
									'fp' : fp,
									'fn' : fn,
									'tp' : tp,
									'accuracy' : accuracy,
									'precision' : precision,
									'specificity' : specificity,
									'recall' : recall,
									'f1' : f1,
									'ppv' : ppv,
									'npv' : npv
									}
	
	return classification_performance


# """
# 	TODO get classification weights 
# """
# def get_classifier_weights(classifier, X_labels):

# 	# get name of the estimator
# 	estimator = classifier.best_estimator_.named_steps['classify'].__class__.__name__
	
# 	# check estimator and print accordingly
# 	if estimator == 'LogisticRegression':

# 		coefficients = classifier.best_estimator_.named_steps['classify'].coef_[0]

# 		for a, b in sorted(zip(X_labels, coefficients), key = lambda x: abs(x[1]), reverse=True ):
# 			print(a, b)

# 	elif estimator == 'DecisionTreeClassifier':

# 		estimator = classifier.best_estimator_.named_steps['classify']

# 		n_nodes = estimator.tree_.node_count
# 		children_left = estimator.tree_.children_left
# 		children_right = estimator.tree_.children_right
# 		feature = estimator.tree_.feature
# 		threshold = estimator.tree_.threshold


