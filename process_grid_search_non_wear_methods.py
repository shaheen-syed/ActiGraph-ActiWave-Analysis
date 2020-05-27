# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import os
import numpy as np
import pandas as pd
import time
import itertools
from multiprocessing import cpu_count, Manager
from joblib import Parallel
from joblib import delayed

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import set_start, set_end, get_subjects_with_invalid_data, calculate_vector_magnitude, save_pickle, load_pickle, create_directory, read_directory, get_current_timestamp, convert_short_code_to_long
from functions.hdf5_functions import get_all_subjects_hdf5, get_datasets_from_group, read_dataset_from_group, save_data_to_group_hdf5
from functions.datasets_functions import get_actigraph_acc_data, get_actigraph_epoch_data, get_actigraph_epoch_60_data
from functions.ml_functions import get_confusion_matrix, calculate_classification_performance, create_train_test_split, return_stratified_k_folds
from functions.plot_functions import plot_grid_search, plot_classification_results_comparison, plot_classification_results_comparison_all
from functions.raw_non_wear_functions import find_consecutive_index_ranges
from algorithms.non_wear_time.hecht_2009 import hecht_2009_triaxial_calculate_non_wear_time
from algorithms.non_wear_time.troiano_2007 import troiano_2007_calculate_non_wear_time
from algorithms.non_wear_time.choi_2011 import choi_2011_calculate_non_wear_time
from algorithms.non_wear_time.hees_2013 import hees_2013_calculate_non_wear_time

"""
	GLOBAL VARIABLES
"""
# local
ACTIGRAPH_HDF5_FILE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIGRAPH_TU7.hdf5')
# ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')
ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join(os.sep, 'users', 'shaheensyed', 'hdf5', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')
# server
# ACTIGRAPH_HDF5_FILE = os.path.join(os.sep, 'media', 'shaheen', 'LaCie_serve', 'ACTIGRAPH_TU7.hdf5')
# ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join(os.sep, 'media', 'shaheen', 'LaCie_serve', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')
# ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join('hdf5', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')

"""
	PROCESS EPOCH FILES
"""
def batch_create_epoch_datasets(use_parallel = True, num_jobs = cpu_count(), limit = None, dataset_prefix = 'epoch'):
	"""
	Create epoch n-seconds datasets, where n could be 10s, 20s, 30s etc. These datasets are pre-created so the grid-search analysis can run faster.
	"""

	# get all the subjects from the hdf5 file and remove subjects with invalid data
	subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]

	# seconds of epoch data, e.g. 10s epoch, 20s epoch
	S = range(10,61,10)

	if use_parallel:

		# verbose
		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(create_epoch_datasets)(subject = subject, S = S, dataset_prefix = dataset_prefix,  idx = i, total = len(subjects)) for i, subject in enumerate(subjects))
		# execute task
		executor(tasks)

	else:

		# loop over all the subjects and perform sensitivity analysis
		for idx, subject in enumerate(subjects):

			create_epoch_datasets(subject, S, dataset_prefix, idx, len(subjects))

def create_epoch_datasets(subject, S, dataset_prefix, idx = 1, total = 1):

	logging.info('{style} Processing subject: {} {}/{} {style}'.format(subject, idx, total, style = '='*10))

	# get actigraph start and stop time
	start_time, stop_time = _get_actigraph_start_stop(subject)
	
	# seconds of epoch data
	for s in S:

		logging.debug('seconds of epoch data : {}'.format(s))

		# get epoch data
		df_epoch_data = _read_epoch_dataset(subject, '{}{}'.format(dataset_prefix, 10), start_time, stop_time, use_vmu = False, upscale_epoch = True, start_epoch_sec = 10, end_epoch_sec = s)

		# check if dataset is not none
		if df_epoch_data is None:
			logging.warning('No epoch data found, skipping...')
			return

		# convert to numpy array and take only XYZ (first three columns)
		epoch_data = df_epoch_data.values[:, :3]
		
		# save to HDF5
		save_data_to_group_hdf5(group = subject, data = epoch_data, data_name = '{}{}'.format(dataset_prefix, s), overwrite = True, create_group_if_not_exists = False, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

"""
	GRID SEARCH
"""
def perform_grid_search(method, nw_method, num_jobs = cpu_count(), save_folder = os.path.join('files', 'grid-search-hees_original')):
	"""
	Perform grid search analysis on epoch or raw data. For epoch data, set method = 'epoch', for raw set method = 'raw'

	Parameters
	-----------
	method : string
		what type of data to process. Options are 'epoch' and 'raw'
	nw_method : string
		which non wear method to use. Options are 'hecht', 'troiano', 'choi', and 'hees'
	num_jobs : int
		number of parallel processes to use to speed up calculation
	save_folder : os.path
		folder location to save classification results to
	"""

	# create list of all possible grid search parameter values combinations
	combinations, *_ = _get_grid_search_parameter_combinations(nw_method)
	
	# get all the subjects from the hdf5 file and remove subjects with invalid data
	subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]

	"""
		LOAD DATA FROM ALL SUBJECTS
	"""

	# empty dictionary to populate with acceleration data
	subjects_data = {x : {'data' : None, 'true_nw_time' : None} for x in subjects}

	# parallel processing
	executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')

	# create tasks so we can execute them in parallel
	tasks = (delayed(_read_epoch_and_true_nw_data)(subject = subject, i = i, total = len(subjects), return_epoch = False if method == 'raw' else True) for i, subject in enumerate(subjects))

	# execute tasks and process the return values
	for subject, subject_data, subject_true_nw in executor(tasks):
		
		# add to dictionary. Note that data will be None if method = 'raw'. See function _read_epoch_and_true_nw_data with return_epoch = False
		subjects_data[subject]['data'] = subject_data
		subjects_data[subject]['true_nw_time'] = subject_true_nw

	"""
		PERFORM GRID SEARCH
	"""

	# keep track of confusion matrix results per combination
	combination_to_confusion_matrix = {x : None for x in combinations}

	# parallel processing of t an i parameters
	executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
	
	# create tasks so we can execute them in parallel
	tasks = (delayed(_calculate_subject_combination_confusion_matrix)(method = method, combination = combination, subjects = subjects, nw_method = nw_method, subjects_data = subjects_data, idx = idx, total = len(combinations)) for idx, combination in enumerate(combinations))
	
	# execute tasks and process the return values
	for combination, cf_matrix in executor(tasks):

		# save classification performance
		combination_to_confusion_matrix[combination] = calculate_classification_performance(*cf_matrix)

	# save classification results to disk
	save_pickle(combination_to_confusion_matrix, 'grid-search-results-{}'.format(nw_method), save_folder)

"""
	CV GRID SEARCH
"""
def perform_cv_grid_search(method, nw_method, num_jobs = cpu_count(), save_folder = os.path.join('files', 'grid-search-cv-hecht')):
	"""
	Perform cross validated grid search

	Parameters
	-----------
	method : string
		what type of data to process. Options are 'epoch' and 'raw'
	nw_method : string
		which non wear method to use. Options are 'hecht', 'troiano', 'choi', and 'hees'
	num_jobs : int
		number of parallel processes to use to speed up calculation
	save_folder : os.path
		folder location to save classification results to
	"""

	# create list of all possible grid search parameter values combinations
	combinations, *_ = _get_grid_search_parameter_combinations(nw_method)
	
	# number of cross validations
	cv = 10
	# train / test split
	split = .3
	# cross validation metric
	cv_metric = 'f1'

	# get all the subjects from the hdf5 file and remove subjects with invalid data
	subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]
 
	# dictionary subject to number of non wear time sequences
	subject_to_nw_sequence = {}

	"""
		LOAD DATA FROM ALL SUBJECTS
	"""

	# empty dictionary to populate with acceleration data
	subjects_data = {x : {'data' : None, 'true_nw_time' : None} for x in subjects}

	# parallel processing
	executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')

	# create tasks so we can execute them in parallel
	tasks = (delayed(_read_epoch_and_true_nw_data)(subject = subject, i = i, total = len(subjects), return_epoch = False if method == 'raw' else True) for i, subject in enumerate(subjects))

	# execute tasks and process the return values
	for subject, subject_data, subject_true_nw in executor(tasks):
		
		# add to dictionary. Note that data will be None if method = 'raw'. See function _read_epoch_and_true_nw_data with return_epoch = False
		subjects_data[subject]['data'] = subject_data
		subjects_data[subject]['true_nw_time'] = subject_true_nw

		# get all indexes with non wear time (nw is encoded as 1, 0 = wear time)
		non_wear_indexes = np.where(subject_true_nw == 1)[0]

		# set class label
		subject_to_nw_sequence[subject] = 0 if len(find_consecutive_index_ranges(non_wear_indexes)) == 1 else 1

	"""
		GET TRAINING AND TEST SET
	"""

	# get list with 0 if no non wear time and 1 if non wear time exists somehwere in the true nw sequence (this can be used to create a stratified split)
	subjects_label = [subject_to_nw_sequence[x] for x in subjects]

	# split subjects into training and testing subjects
	# train_subjects, test_subjects, *_ = create_train_test_split(subjects, subjects_label, test_size = split, shuffle = False, random_state = 42, stratify = subjects_label)
	train_subjects, test_subjects, *_ = create_train_test_split(subjects, subjects_label, test_size = split, shuffle = False)

	"""
		PERFORM CROSS VALIDATION
	"""
	# dictionary to store fold results
	all_fold_results = {x : {'combination' : None, 'training_results' : None, 'test_results' : None} for x in range(cv)}

	# start a manager to share the tracker dictionary accross parallel processed
	manager = Manager()
	# create the tracker as a manager dictionary
	subject_combination_tracker = manager.dict()
	
	# loop over eacht fold
	fold_cnt = 0
	for train_idx, test_idx in return_stratified_k_folds(n_splits = cv).split(train_subjects, [subject_to_nw_sequence[s] for s in train_subjects]):

		# keep track of time it takes to complete one fold
		epoch_tic = time.time()

		logging.info('{style} Processing Fold : {} {style}'.format(fold_cnt + 1, style = '='*10))

		# get the training subjects part of the fold
		train_fold_subjects = [train_subjects[x] for x in train_idx]
		# get the test subjects of the fold
		test_fold_subjects = [train_subjects[x] for x in test_idx]

		# keep track of confusion matrix results per combination
		combination_to_confusion_matrix = {x : None for x in combinations}

		# parallel processing of t an i parameters
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		
		# create tasks so we can execute them in parallel
		tasks = (delayed(_calculate_subject_combination_confusion_matrix)(method = method, combination = combination, subjects = train_fold_subjects, nw_method = nw_method, \
				subject_combination_tracker = subject_combination_tracker, subjects_data = subjects_data, idx = idx, total = len(combinations)) for idx, combination in enumerate(combinations))
		
		# execute tasks and process the return values
		for combination, cf_train in executor(tasks):

			combination_to_confusion_matrix[combination] = calculate_classification_performance(*cf_train)
			
			
		logging.debug('-\tItems in combination tracker: {}'.format(len(subject_combination_tracker)))

		# find combination with the highest accuracy 
		top_combination = sorted(combination_to_confusion_matrix.items(), key = lambda item: item[1][cv_metric], reverse = True)[0]

		# apply top combination on test subjects
		_, cv_confusion_test = _calculate_subject_combination_confusion_matrix(method = method, combination = top_combination[0], subjects = test_fold_subjects, nw_method = nw_method, \
						subject_combination_tracker = subject_combination_tracker, subjects_data = subjects_data)

		# save fold results
		all_fold_results[fold_cnt]['combination'] = top_combination[0]
		all_fold_results[fold_cnt]['training_results'] = top_combination[1]
		all_fold_results[fold_cnt]['test_results'] = calculate_classification_performance(*cv_confusion_test)
		
		# verbose
		logging.info('-\ttop combination: {}'.format(all_fold_results[fold_cnt]['combination']))
		logging.info('-\ttraining results: {}'.format(all_fold_results[fold_cnt]['training_results']))
		logging.info('-\ttest results: {}'.format(all_fold_results[fold_cnt]['test_results']))
		logging.info('-\texecuted fold in {} seconds'.format(time.time() - epoch_tic))

		# increase fold counter
		fold_cnt += 1

	# get combination of folds with best accuracy
	top_cv_combination = None 
	top_cv_metric = 0

	for value in all_fold_results.values():
		if value['test_results'][cv_metric] > top_cv_metric:
			top_cv_metric = value['test_results'][cv_metric]
			top_cv_combination = value['combination']
		
	logging.info('Top combination training: {}, {}: {}'.format(top_cv_combination, cv_metric, top_cv_metric))

	# obtain training classification results
	combined_training_results = {'accuracy': [], 'precision': [], 'specificity': [], 'recall': [], 'f1': [], 'ppv': [], 'npv': []}
	for training_results in all_fold_results.values():
		
		for result_key in combined_training_results.keys():

			combined_training_results[result_key].append(training_results['test_results'][result_key])
	
	# calculate average of classification scores
	for key, value in combined_training_results.items():
		combined_training_results[key] = np.nanmean(value)

	logging.info('='*60)	
	logging.info('{}-Fold cross validation Training results: {}'.format(cv, combined_training_results))	

	# try best combination obtained from cross validation on test subjects
	_, confusion_test = _calculate_subject_combination_confusion_matrix(method = method, combination = top_cv_combination, subjects = test_subjects, nw_method = nw_method, subjects_data = subjects_data)

	# get test classification performance
	test_results = calculate_classification_performance(*confusion_test)

	logging.info('{}-Fold cross validation Test results: {}'.format(cv, test_results))	

	# classification results
	classification_data = {	'combination' : top_cv_combination, 
							'training' : all_fold_results,
							'combined_training' : combined_training_results,
							'test' : test_results}
	
	# save classification results to disk
	save_pickle(classification_data, 'cv-grid-search-results-{}'.format(nw_method), save_folder)

	# save tracker
	save_pickle(dict(subject_combination_tracker), 'cv-grid-search-tracker-{}'.format(nw_method), save_folder)
	

"""
	PLOT GRID SEARCH
"""
def perform_plot_grid_search(nw_method, data_folder = os.path.join('files', 'grid-search', 'final_reverse_prec_rec_new_hecht')):

	# load the classification data
	data = load_pickle('grid-search-results-{}'.format(nw_method), data_folder)

	# define classification metric to plot
	classifications = ['accuracy', 'precision', 'recall', 'f1']

	# start to plot each classification
	for classification in classifications:

		# dynamically create plot name
		plot_name = 'grid_search_plot_{}_{}.pdf'.format(nw_method, classification)
		# plot folder
		plot_folder = os.path.join('plots', 'paper')

		# skip parameters
		skip_parameters, skip_combinations = [], []
		# overwrite top values
		overwrite = {} 

		# skip parameter values completely
		if nw_method == 'hecht':
			
			# define plot parameters
			plot_parameters = {	'num_rows' : 1,
								'num_columns' : 3,
								'figsize' : (9,3),
								'annotations' : True,
								'vmin' : 0,
								'vmax' : .3,
								'levels' : 11
								}

		if nw_method == 'troiano':
			
			skip_parameters = ['VM']
			skip_combinations = ['AT_ST']	
			
			# overwrite = {'AT' : '50','ST' : '4'}

			# define plot parameters
			plot_parameters = {	'num_rows' : 2,
								'num_columns' : 3,
								'figsize' : (9,6),
								'annotations' : True,
								'vmin' : 0,
								'vmax' : 1.,
								'levels' : 9,
								'remove_plots' : [-1]
								}

		if nw_method == 'choi':

			skip_parameters = ['VM', 'AT']

			# skip_combinations = ['ST_MWL']
			# overwrite = {'ST' : '2'}

			# define plot parameters
			plot_parameters = {	'num_rows' : 2,
								'num_columns' : 3,
								'figsize' : (9,6),
								'annotations' : True,
								'vmin' : 0,
								'vmax' : 1.,
								'levels' : 11,
								# 'remove_plots' : [-1]
								}
		
		if nw_method == 'hees':

			skip_parameters = ['WO']
			# skip_combinations = ['ST_VT']
			# skip_combinations = ['VT_VA']

			# define plot parameters
			plot_parameters = {	'num_rows' : 4,
								'num_columns' : 3,
								'figsize' : (9,12),
								'annotations' : True,
								'vmin' : 0,
								'vmax' : 1.,
								'levels' : 8,
								'remove_plots' : [-2, -1]
								}

		# get grid search combinations, with parameter and labels
		_, parameters, labels, default_parameters  = _get_grid_search_parameter_combinations(nw_method)

		# empty dictionary to store plot data
		plot_data, annotations = _get_plot_data(data, parameters, default_parameters, classification, skip_parameters, skip_combinations, overwrite)

		# call plot function
		plot_grid_search(plot_data, nw_method, classification, labels, annotations, plot_parameters, plot_name, plot_folder)



"""
	OTHER PLOTS
"""
def perform_plot_comparison_default_optimized(folder_results = os.path.join('files', 'grid-search', 'final_reverse_prec_rec_new_hecht')):

	# results of grid search
	hecht_data = load_pickle(file_name = 'grid-search-results-hecht.pkl', folder = folder_results)
	troiano_data = load_pickle(file_name = 'grid-search-results-troiano.pkl', folder = folder_results)
	choi_data = load_pickle(file_name = 'grid-search-results-choi.pkl', folder = folder_results)
	hees_data = load_pickle(file_name = 'grid-search-results-hees.pkl', folder = folder_results)
	
	"""
		DEFAULT VARIABLES
	"""

	# hecht: T: threshold value in VMU (5), I: time intervals in minutes (20), M : min count (2)
	hecht_default = '5-20-2'
	# troiano default: AT activity threshold (default :activity_threshold = 0), MPL minimum period length (default : min_period_len = 60), ST spike tolerance (default : spike_tolerance = 2)
	# SS spike stoplevel (default : spike_stoplevel = 100), VM use vector magnitude (default : use_vector_magnitude = False)
	troiano_default = '0-60-2-100-False'
	# AT activity threshold (default = 0), MPL minimum period length (default = 90), ST spike tolerance (default = 2)
	# MWL minimum window length (default = 30), WST window_spike_tolerance = 0,  VM use vector magnitude (default = False)
	choi_default = '0-90-2-30-0-False'
	# MW minimum non wear time window (default = 60), WO window overlap (default = 15). ST standard deviation threshold (default 3.0)
	# SA standard deviation minimum number of axes (default 2), VT value range threshold (default 50.), VA value range min number of axes (default 2)
	hees_default = '60-15-3-2-50-2'

	# define what classification metrics to plot
	classifications = ['accuracy', 'precision', 'recall', 'f1']

	# empty dataframe
	df = pd.DataFrame()
	df_all = pd.DataFrame()

	# get data for each type of classification
	for classification in classifications:

		"""
			OPTIMIZED VALUE
		"""
		hecht_optimized_value = sorted(hecht_data.items(), key = lambda item: item[1][classification], reverse = True)[0][1]
		troiano_optimized_value = sorted(troiano_data.items(), key = lambda item: item[1][classification], reverse = True)[0][1]
		choi_optimized_value = sorted(choi_data.items(), key = lambda item: item[1][classification], reverse = True)[0][1]
		hees_optimized_value = sorted(hees_data.items(), key = lambda item: item[1][classification], reverse = True)[0][1]

		# combined data
		data = {'Hecht' : [hecht_data[hecht_default][classification], hecht_optimized_value[classification]],
				'Troiano' : [troiano_data[troiano_default][classification], troiano_optimized_value[classification]],
				'Choi' : [choi_data[choi_default][classification], choi_optimized_value[classification]],
				'Hees' : [hees_data[hees_default][classification], hees_optimized_value[classification]]
				}

		# all data
		data_all = {'Hecht' : [hecht_data[hecht_default], hecht_optimized_value],
				'Troiano' : [troiano_data[troiano_default], troiano_optimized_value],
				'Choi' : [choi_data[choi_default], choi_optimized_value],
				'Hees' : [hees_data[hees_default], hees_optimized_value]
				}

		df[classification] = pd.Series(data)
		df_all[classification] = pd.Series(data_all)

	# plot_classification_results_comparison(df.T)
	plot_classification_results_comparison_all(df_all)



"""
	INTERNAL HELPER FUNCTIONS
"""

def _get_plot_data(data, parameters, default_parameters, classification, skip_parameters = None, skip_combinations = None, overwrite = None, fix_parameter_to_default = True):
	"""
	Create plot data for combinations of two parameters. For instance, create the dataframe comparing grid search values for AT and MLP

	Parameters
	----------
	data : dictionary
		classification data. Key is a combination of parameter values with hyphens, for example 90-20-0-10-False, the value contains all the classification values (for example accuracy, precision, recall, f1)
	parameters :  dictionary
		key = Grid search variables names, values are the variable values. For example 'AT' : [0,10,20,30]
	classification : string
		type of classification to plot. Can be one of the following: 'tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'specificity', 'recall', 'f1', 'ppv', 'npv'
	skip_parameters : list (optional)
		list of parameter values that do not need to be plot. For example 'VM' for vector magnitude and this will results in VM not being compared to any other variable
	skip_combinations : list  (optional)
		skip a specific combination from plotting. Almost similar to skip_parameters but now a specific combination will be ignored, for example AT_MWL. Other combinations with AT or MWL will still be plotted
	overwrite : dictionary (optional)
		key = parameter name, value = parameter value that needs to be fixed. For example, different values for the same parameter can all achieve a top classification score, here one needs to be fixed, which one can be specified here.

	Returns
	----------
	plot_data : dictinary
		key = grid search combination (for example '90-20-0-10'), value = dataframe with classification values
	annotations : dictionary
		annotation to show parameter value that obtains the highest classification value. key = parameter value (for example 'AT'), value = parameter value (for example 10) 
	"""

	# handle optional arguments
	if skip_parameters is None : skip_parameters = []
	if skip_combinations is None: skip_combinations = []
	if overwrite is None: overwrite = {}

	# give each parameter an index
	par_index = dict(zip(parameters, range(len(parameters))))

	# find values that produce the largest classification score (accuracy, precision, recall, f1)
	top_results = sorted(data.items(), key = lambda item: item[1][classification], reverse = True)[0]
	
	# empty dictionary to store dataframes to
	plot_data = {}

	# loop over combinations of parameter types, for instance MW and ST
	for i, first_par in enumerate(list(parameters.keys())):
		for second_par in list(parameters.keys())[i+1:]:

			# skip parameters that you dont want to plot
			if first_par in skip_parameters or second_par in skip_parameters:
				continue

			# create key, for example 'WM_ST', this will then hold the plot data of comparison WM and ST (while fixing the remaining variables)
			plot_key = '{}_{}'.format(first_par, second_par)

			if plot_key in skip_combinations:
				continue
			# add combination to dataframe
			plot_data[plot_key] = pd.DataFrame()

			# loop over first parameter
			for row in parameters[first_par]:
				# empty list that can hold the row data
				row_data = []
				# loop over the second parameter
				for column in parameters[second_par]:
				
					# get variables from top result (here we only have to change the row and column variable)
					plot_results = top_results[0].split('-')
					
					# overwrite the row index
					plot_results[par_index[first_par]] = row
					# overwrite index that represents the column value
					plot_results[par_index[second_par]] = column

					# if set to True, then change fixing parameter to default value, instead of fixing it to the values that result in the best classification score
					if fix_parameter_to_default:
						# keep remaining values fixed to their default values
						for fixed_par in parameters.keys():
							# fixed parameter cannot be the parameter that we use on the x or y axis for the contour plots
							if fixed_par not in [first_par,second_par]:
								
								# get the default value of the fixed parameter.
								default_value = default_parameters[fixed_par]
								
								# change optimized parameter with the default parameter at the right location (obtained by looking at par_index)
								plot_results[par_index[fixed_par]] = default_value
						
					# overwrite values (but not if part of combination)
					for overwrite_key, overwrite_value in overwrite.items():
						if overwrite_key not in plot_key:
							plot_results[par_index[overwrite_key]] = overwrite_value

					# get the classification value
					c_value = data['-'.join([str(x) for x in plot_results])][classification]

					# add to row data (so we can append it later as a series)
					row_data.append(c_value)
				
				# add the row data as a series to the dataframe
				plot_data[plot_key][row] = pd.Series(row_data, index = parameters[second_par])
	
	# # annotations for top result
	# annotations = {}
	# for par in parameters.keys():
	# 	if par not in skip_parameters:
	# 		annotations[par] = int(top_results[0].split('-')[par_index[par]])

	return plot_data, default_parameters

def _get_hecht_grid_search_nw_vector(variables, data, reverse = True, s = 60, verbose = False):

	# unpack variables
	t, i, m = variables

	if verbose:
		logging.debug('Calculating t: {}, i: {}, m: {}'.format(*variables))

	# calculate the VMU
	data = calculate_vector_magnitude(data)

	# retrieve non-wear vector
	nw_vector = hecht_2009_triaxial_calculate_non_wear_time(data = data, epoch_sec = s, threshold = int(t), time_interval_mins = int(i), min_count = int(m))

	# upscale nw_vector
	nw_vector = nw_vector.repeat(s, axis = 0)

	if reverse:
		# reverse 0>1 and 1>0 (the reason we do this is that now the positive examples are the non wear times. )
		nw_vector = 1 - nw_vector

	return nw_vector, t, i, m

def _get_troiano_grid_search_nw_vector(variables, data, reverse = True, verbose = False):

	# unpack variables
	at, mpl, st, ss, vm = variables

	if verbose:
		logging.debug('Calculating at: {}, mpl: {}, st: {}, ss: {}, vm: {}'.format(*variables))

	# obtain Troiano non wear vector
	nw_vector = troiano_2007_calculate_non_wear_time(data, None, activity_threshold = int(at), min_period_len = int(mpl), spike_tolerance = int(st), spike_stoplevel = int(ss), use_vector_magnitude = eval(vm), print_output = False)

	# upscale nw_vector
	nw_vector = nw_vector.repeat(60, axis = 0)

	# flip 0>1 and 1>0
	if reverse:
		nw_vector = 1 - nw_vector

	return nw_vector, at, mpl, st, ss, vm

def _get_choi_grid_search_nw_vector(variables, data, reverse = True, verbose = False):
	"""

	"""

	# unpack variables
	at, mpl, st, mwl, wst, vm = variables

	if verbose:
		logging.debug('Calculating at: {}, mpl: {}, st: {}, mwl: {}, vm: {}'.format(*variables))

	# obtain Choi non wear vector
	nw_vector = choi_2011_calculate_non_wear_time(data, None, activity_threshold = int(at), min_period_len = int(mpl), spike_tolerance = int(st),  min_window_len = int(mwl), window_spike_tolerance = int(wst), use_vector_magnitude = eval(vm), print_output = False)

	# upscale nw_vector
	nw_vector = nw_vector.repeat(60, axis = 0)

	# flip 0>1 and 1>0
	if reverse:
		nw_vector = 1 - nw_vector

	return nw_vector, at, mpl, st, mwl, wst, vm

def _get_hees_grid_search_nw_vector(variables, data, reverse = True, hz = 100):
	"""

	"""
	 
	# unpack variables
	mw, wo, st, sa, vt, va = variables

	# obtain Troiano non wear vector
	nw_vector = hees_2013_calculate_non_wear_time(data, hz = hz, min_non_wear_time_window = int(mw), window_overlap = int(wo), std_mg_threshold = float(st), std_min_num_axes = int(sa) , value_range_mg_threshold = float(vt), value_range_min_num_axes = int(va))

	# downscale nw_vector to 1s samples (so it can be compared to true non wear time that is on 1s resolution)
	nw_vector = nw_vector[::100]
	
	# flip 0>1 and 1>0
	if reverse:
		nw_vector = 1 - nw_vector

	return nw_vector, mw, wo, st, sa, vt, va

def _get_classification_performance(key, y, temp_folder, subject_order):

	logging.debug('Calculating {}'.format(key))

	# read y_hat from file
	y_hat_from_file = []

	# loop over subjects in the order in which they were processed
	for s in subject_order:

		# create the name of the folder dynamically where data is stored
		folder_name = os.path.join(temp_folder, key)

		# load the grid search data
		grid_data = load_pickle(s, folder_name)

		# add to list
		y_hat_from_file.append(grid_data)
	
	# convert list of arrays to one new array by vertically stacking them
	y_hat = np.vstack(y_hat_from_file)

	# get confusion matrix values
	tn, fp, fn, tp = get_confusion_matrix(y, y_hat, labels = [0,1]).ravel()

	# calculate classification performance such as precision, recall, f1 etc.
	classification_performance = calculate_classification_performance(tn, fp, fn, tp)

	# verbose
	logging.debug(classification_performance)

	return key, classification_performance

def _get_actigraph_start_stop(subject):
	"""
	Returns the start and stop time of overlapping actigraph and actiwave data. 

	Parameters
	---------
	subject : string
		subject ID

	Returns
	--------
	start_time : np.datetime64
		start time of the actigraph-actiwave mapping signal
	stop_time : np.datetime64
		stop time of the actigraph-actiwave mapping signal

	Example
	--------
	>>start_time
	numpy.datetime64('2015-07-04T00:00:00.000')
	>>stop_time
	numpy.datetime64('2015-07-04T13:11:21.990')
	"""

	# read actigraph acceleration time
	_, _, actigraph_time = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	
	# get start and stop time
	start_time, stop_time = actigraph_time[0], actigraph_time[-1]

	return start_time, stop_time

def _read_epoch_dataset(subject, epoch_dataset, start_time, stop_time, use_vmu = False, upscale_epoch = True, start_epoch_sec = 10, end_epoch_sec = 60):
	"""
	Read epoch dataset

	Parameters
	----------
	subject : string
		subject ID
	epoch_dataset : string
		name of HDF5 dataset that contains the epoch data

	Returns
	----------
	"""

	# check if epoch dataset is part of HDF5 group
	if epoch_dataset in get_datasets_from_group(group_name = subject, hdf5_file = ACTIGRAPH_HDF5_FILE):

		# get actigraph 10s epoch data
		epoch_data, _ , epoch_time_data = get_actigraph_epoch_data(subject, epoch_dataset = epoch_dataset, hdf5_file = ACTIGRAPH_HDF5_FILE)

		# if upscale_epoch is set to True, then upsample epoch counts to epoch_sec
		if upscale_epoch:
				
			# convert to 60s epoch data	
			epoch_data, epoch_time_data = get_actigraph_epoch_60_data(epoch_data, epoch_time_data, start_epoch_sec, end_epoch_sec)

		# if set to true, then calculate the vector magnitude of all axes
		if use_vmu:

			# calculate epoch VMU
			epoch_data = calculate_vector_magnitude(epoch_data[:,:3], minus_one = False, round_negative_to_zero = False)

		# create dataframe of actigraph acceleration 
		df_epoch_data = pd.DataFrame(epoch_data, index = epoch_time_data).loc[start_time:stop_time]
	
		return df_epoch_data

	else:
		logging.warning('Subject {} has no {} dataset'.format(subject, epoch_dataset))
		return None

def _read_epoch_and_true_nw_data(subject, i = 1, total = 1, return_epoch = True, return_true = True):

	logging.info('Loading subject {} into memory {}/{}'.format(subject, i, total))

	"""
		EPOCH DATA
	"""
	if return_epoch:
		# read subject epoch data
		subject_epoch_data = read_dataset_from_group(group_name = subject, dataset = 'epoch60', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

		# lower precision
		subject_epoch_data = subject_epoch_data.astype('float16')
	else:
		# read raw data and downscale to 1s
		# subject_data, *_ = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
		# subject_data = subject_data[::100]
		# subject_epoch_data = subject_data

		# set data to none, no epoch data is returned. 
		subject_epoch_data = None

	"""
		TRUE NON WEAR TIME
	"""
	if return_true:
		# read true non wear time and convert 0>1 and 1->0
		subject_true_nw = 1 - read_dataset_from_group(group_name = subject, dataset = 'actigraph_true_non_wear', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE).astype('uint8').reshape(-1,1)
		# convert to 1s instead of 100hz
		subject_true_nw = subject_true_nw[::100]
	else:
		subject_true_nw = None

	return subject, subject_epoch_data, subject_true_nw

def _get_grid_search_parameter_combinations(nw_method):
	"""
	Get combinations of grid search parameters for non wear methods
	"""

	if nw_method == 'hecht':

		# threshold value in VMU
		T = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # 10X
		# time intervals in minutes
		I = range(5, 100 + 1, 5) # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
		# min count
		M = [1, 2, 3, 4, 5] # 5x

		# create list of all possible combinations
		combinations = [f'{t}-{i}-{m}' for t in T for i in I for m in M]

		# parameters
		parameters = {'T' : T, 'I' : I, 'M' : M}

		# default parameter values
		default_parameters = {'T' : 5, 'I' : 20, 'M' : 2}

		# get labels for each parameter
		labels = {key : convert_short_code_to_long(key) for key in parameters.keys()}

	elif nw_method == 'troiano':

		# activity threshold (default :activity_threshold = 0)
		AT = [0, 25, 50, 75, 100]
		# minimum period length (default : min_period_len = 60)
		MPL = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
		# spike tolerance (default : spike_tolerance = 2)
		ST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		# spike stoplevel (default : spike_stoplevel = 100)
		SS = [1, 25, 50, 75, 100, 125, 150, 175, 200]
		# use vector magnitude (default : use_vector_magnitude = False)
		VM = [True, False] # 2x

		combinations = [f'{at}-{mpl}-{st}-{ss}-{vm}' for at in AT for mpl in MPL for st in ST for ss in SS for vm in VM]

		parameters = {'AT' : AT, 'MPL' : MPL, 'ST' : ST, 'SS' : SS, 'VM' : VM}

		# troiano default parameters '0-60-2-100-False'
		default_parameters = {'AT' : 0, 'MPL' : 60, 'ST' : 2, 'SS' : 100, 'VM' : False}

		# get labels for each parameter
		labels = {key : convert_short_code_to_long(key) for key in parameters.keys()}

	elif nw_method == 'choi':

		# activity threshold (default = 0)
		AT = [0, 25, 50, 75, 100]
		# minimum period length (default = 90)
		MPL = [30, 60, 90, 120, 150, 180, 210]
		# spike tolerance (default = 2)
		ST = [1, 2, 3, 4, 5] # 6x
		# minimum window length (default = 30)
		MWL = [10, 20, 30, 40, 50, 60] # 6x
		# number of spikes allowed in the second window (default = window window_spike_tolerance = 0)
		WST = [0, 1, 2, 3, 4] # 5x
		# use vector magnitude (default = False)
		VM = [True, False] # 2x

		combinations = [f'{at}-{mpl}-{st}-{mwl}-{wst}-{vm}' for at in AT for mpl in MPL for st in ST for mwl in MWL for wst in WST for vm in VM]

		parameters = {'AT' : AT, 'MPL' : MPL, 'ST' : ST, 'MWL' : MWL, 'WST' : WST, 'VM' : VM }

		# default parameter '0-90-2-30-0-False'
		default_parameters = {'AT' : 0, 'MPL' : 90, 'ST' : 2, 'MWL' : 30, 'WST' : 0, 'VM' : False }

		# get labels for each parameter
		labels = {key : convert_short_code_to_long(key) for key in parameters.keys()}

	elif nw_method == 'hees':

		# minimum non wear time window (default = 60)
		MW = [15, 30, 45, 60, 75, 90, 105, 120, 135]
		# window overlap (default = 15)
		# WO = [5, 10, 15, 20, 25, 30]# 6x
		WO = [1]
		# standard deviation threshold (default 3.0)
		ST = [2, 3, 4, 5, 6, 7, 8]
		# standard deviation minimum number of axes (default 2)
		SA = [1, 2, 3]
		# value range threshold (default 50.)
		VT = [1, 5, 10, 15, 20, 25, 50, 75, 100]
		# value range min number of axes (default 2)
		VA = [1, 2, 3]

		"""
		Default combinations and optimized combinations for testing
		"""
		# MW = [135]
		# WO = [1]
		# ST = [8]
		# SA = [2]
		# VT = [1]
		# VA = [1]
		

		combinations = [f'{mw}-{wo}-{st}-{sa}-{vt}-{va}' for mw in MW for wo in WO for st in ST for sa in SA for vt in VT for va in VA]

		parameters = {'MW' : MW, 'WO' : WO, 'ST' : ST, 'SA' : SA, 'VT' : VT, 'VA' : VA}

		# default paramters '60-15-3-2-50-2'
		default_parameters = {'MW' : 60, 'WO' : 15, 'ST' : 3, 'SA' : 2, 'VT' : 50, 'VA' : 2}

		# define labels
		labels = {	'MW' : 'Minimum interval (mins)', 
					'WO' : 'Window overlap',
					'ST' : 'Std. threshold (mg)',
					'SA' : 'Std. min. #axes',
					'VT' : 'Value threshold (mg)',
					'VA' : 'Value min. #axes'
				}

	else:
		logging.error('Non wear method not implemented: {}'.format(nw_method))
		exit()

	return combinations, parameters, labels, default_parameters

def _calculate_subject_combination_confusion_matrix(method, combination, subjects, nw_method, subject_combination_tracker = None, subjects_data = None, verbose = True, idx = 1, total = 1):

	# verbose
	if verbose:
		logging.info('Processing {}'.format(combination.split('-')))
		logging.info('Total number of subjects {}'.format(len(subjects)))
	if idx % 100 == 0 and idx != 0:
		logging.debug('-\tProcessed combinations {}/{}'.format(idx, total))
	
	# empty list to store classification data to
	classification_data = []

	# process training fold subjects
	for i, subject in enumerate(subjects):

		if verbose:
			logging.info(f'Processing subject {i}/{len(subjects)}')

		"""
		CHECK IF SUBJECT COMBINATION HAS ALREADY BEEN PROCESSED
		"""
		if subject_combination_tracker is None or subject_combination_tracker.get('{}-{}'.format(subject, combination)) is None:

			"""
				GET ACCELERATION DATA
			"""
			if method == 'epoch':
				# get epoch data
				subject_data = subjects_data[subject]['data']
				# get true non wear time
				subject_true_nw = subjects_data[subject]['true_nw_time']
			elif method == 'raw':
				# get raw data
				subject_data, *_ = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
				
				# get raw data from dictionary
				# subject_data = subjects_data[subject]['data']
				
				# get true non wear time
				subject_true_nw = subjects_data[subject]['true_nw_time']
			else:
				logging.error('Method not implemented: {}'.format(method))
				exit()


			"""
				GET NW DATA
			"""

			if nw_method == 'hecht':
				# get non-wear vector
				subject_nw_vector, *_ = _get_hecht_grid_search_nw_vector(variables = combination.split('-'), data = subject_data)
			elif nw_method == 'troiano':
				# get non-wear vector
				subject_nw_vector, *_ = _get_troiano_grid_search_nw_vector(variables = combination.split('-'), data = subject_data)
			elif nw_method == 'choi':
				# get non-wear vector
				subject_nw_vector, *_ = _get_choi_grid_search_nw_vector(variables = combination.split('-'), data = subject_data)
			elif nw_method == 'hees':
				# get non-wear vector
				subject_nw_vector, *_ = _get_hees_grid_search_nw_vector(variables = combination.split('-'), data = subject_data)#, hz=1)
			else:
				logging.error('Non-wear method {} not defined.'.format(nw_method))
				exit(1)

			"""
				CLIPPING
			"""
			# check lengths of nw_vector and subject_true_nw; they need to be of the same length otherwise the mapping later on will fail
			if subject_nw_vector.shape[0] != subject_true_nw.shape[0]:
			
				# clip on on smallest length to make two vectors equal
				min_clip_length = min(subject_nw_vector.shape[0], subject_true_nw.shape[0])

				# clip
				subject_true_nw = subject_true_nw[:min_clip_length]
				subject_nw_vector = subject_nw_vector[:min_clip_length]

			"""
				CONFUSION MATRIX
			"""

			# get confusion matrix values
			tn, fp, fn, tp = get_confusion_matrix(subject_true_nw, subject_nw_vector, labels = [0,1]).ravel()

			# add to tracker
			if subject_combination_tracker is not None:
				subject_combination_tracker['{}-{}'.format(subject, combination)] = [tn, fp, fn, tp]

		else:
			
			# read classification results from tracker
			tn, fp, fn, tp = subject_combination_tracker['{}-{}'.format(subject, combination)]
		
		# return confusion matrix results
		classification_data.append([tn, fp, fn, tp])
			
	# combine data into numpy array and calculate column totals
	classification_data = np.sum(np.array(classification_data), axis = 0)

	return combination, classification_data


if __name__ == '__main__':

	# start timer and memory counter
	tic, process, logging = set_start()

	# 1) prepare 10s, 20s, 30s, 50s, and 60s epoch data (this will make it faster to run grid search analysis since the epoch data is precomputed)
	# batch_create_epoch_datasets()

	"""
		2) perform grid search on all subjects
	"""
	# perform_grid_search(method = 'epoch', nw_method = 'hecht')
	# perform_grid_search(method = 'epoch', nw_method = 'troiano')
	# perform_grid_search(method = 'epoch', nw_method = 'choi')
	perform_grid_search(method = 'raw', nw_method = 'hees')

	"""
		3) perform cross validated grid search
	"""
	# perform_cv_grid_search(method = 'epoch', nw_method = 'hecht')
	# perform_cv_grid_search(method = 'epoch', nw_method = 'troiano')
	# perform_cv_grid_search(method = 'epoch', nw_method = 'choi')
	# perform_cv_grid_search(method = 'raw', nw_method = 'hees')

	"""
		4) plot grid search analysis contourplot
	"""
	# perform_plot_grid_search(nw_method = 'hecht')
	# perform_plot_grid_search(nw_method = 'troiano')
	# perform_plot_grid_search(nw_method = 'choi')
	# perform_plot_grid_search(nw_method = 'hees')

	"""
		OTHER PLOTS
	"""
	
	# perform_plot_comparison_default_optimized()


	set_end(tic, process)