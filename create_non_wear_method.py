# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import os
import re
import numpy as np
import pandas as pd
import datetime
import time
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from scipy import stats
from sklearn.utils import shuffle

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import 	set_start, set_end, load_yaml, read_directory, create_directory, get_subjects_with_invalid_data, save_pickle, load_pickle,\
										calculate_vector_magnitude, get_subject_counters_for_correction, get_subject_counter_to_merge, get_random_number_between,\
										delete_directory
from functions.datasets_functions import get_actigraph_acc_data
from functions.hdf5_functions import save_data_to_group_hdf5, get_all_subjects_hdf5, read_dataset_from_group, get_datasets_from_group
from functions.raw_non_wear_functions import find_candidate_non_wear_segments_from_raw, find_consecutive_index_ranges, group_episodes
from functions.plot_functions import plot_merged_episodes, plot_cnn_classification_performance, plot_training_results_per_epoch, plot_cnn_inferred_nw_time, plot_episodes_used_for_training, plot_baseline_performance, plot_performance_cnn_nw_method, plot_episodes_used_for_training_combined
from functions.datasets_functions import get_actigraph_acc_data, get_actiwave_acc_data, get_actiwave_hr_data
from functions.ml_functions import get_confusion_matrix, calculate_classification_performance
from functions.dl_functions import create_1d_cnn_non_wear_episodes, load_tf_model


"""
	PREPROCESSING
"""

def merge_close_episodes_from_training_data(read_folder, save_folder):
	"""
	Merge epsisodes that are close to each other. Two close episodes could for instance have some artificial movement between them.
	Note that this merging is done on our labeled gold standard dataset, and this merging is only for training purposes.

	Note that merge and group are used here interchangably

	Parameters
	-----------
	label_folder : os.path()
		folder location of start and stop labels for non-wear time
	save_folder : os.path()
		folder where to save the merged episodes to
	"""

	# read all files to process
	F = [f for f in read_directory(read_folder) if f[-4:] == '.csv']

	# process each file
	for f_idx, f in enumerate(F):

		# extract subject from file
		subject = re.search('[0-9]{8}', f)[0]

		logging.info(f'=== Processing subject {subject}, file {f_idx}/{len(F)} ===')

		# read csv file as dataframe
		episodes = pd.read_csv(f)

		# empty dataframe for grouped episodes
		grouped_episodes = pd.DataFrame()

		# check if episodes is not empty
		if not episodes.empty:
			
			"""
				MERGE NON-WEAR EPISODES
			"""
			
			# group the following episodes (note that we work with counters here, so first episode is counter 0, second episode is counter 1 etc)
			group_nw_episodes = get_subject_counter_to_merge(subject)
			# empty list for episodes to group
			group_nw_episodes_list = [] if group_nw_episodes is None else range(group_nw_episodes[0], group_nw_episodes[1] + 1)

			# start grouping nw episodes based on get_subject_counter_to_merge
			if group_nw_episodes is not None:

				# create the combination of counters, for example '1-4'
				counter_label = f'{group_nw_episodes[0]}-{group_nw_episodes[1]}'

				# grouped non-wear time
				grouped_episodes[counter_label] = pd.Series({	'counter' : counter_label,
																'start' : episodes.iloc[group_nw_episodes[0]]['start'],
																'start_index' : episodes.iloc[group_nw_episodes[0]]['start_index'],
																'stop' : episodes.iloc[group_nw_episodes[1]]['stop'],
																'stop_index' : episodes.iloc[group_nw_episodes[1]]['stop_index'],
																'label' : episodes.iloc[group_nw_episodes[0]]['label']})
			
			# add non-wear time that not need be grouped
			for _, row in episodes[episodes['label'] == 0].iterrows():
				if row.loc['counter'] not in group_nw_episodes_list:
					# save to new dataframe
					grouped_episodes[row.loc['counter']] = pd.Series({ 'counter' : row.loc['counter'],
															'start_index' : row.loc['start_index'],
															'start' : row.loc['start'],
															'stop_index' : row.loc['stop_index'],
															'stop' : row.loc['stop'],
															'label' : row.loc['label'],
															})

			"""
				GROUP WEAR EPISODES
			"""
			
			grouped_wear_episodes = group_episodes(episodes = episodes[episodes['label'] == 1], distance_in_min = 3, correction = 3, hz = 100, training = True)

			"""
				COMBINE TWO DATAFRAMES
			"""
			
			if not grouped_wear_episodes.empty:
				# combine two dataframes
				grouped_episodes = pd.concat([grouped_episodes, grouped_wear_episodes], axis=1, sort = True)

		# create the save folder if not exists
		create_directory(save_folder)
		# save to file + transpose
		grouped_episodes.T.to_csv(os.path.join(save_folder, f'{subject}.csv'))

def process_calculate_true_nw_time_from_labeled_episodes(merged_episodes_folder, hdf5_read_file, hdf5_save_file, std_threshold = 0.004):

	# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
	subjects = get_all_subjects_hdf5(hdf5_file = hdf5_read_file)

	# exclude subjects that have issues with their data
	subjects = [s for s in subjects if s not in get_subjects_with_invalid_data()]

	# loop over each subject
	for i, subject in enumerate(subjects):

		logging.info(f'=== Processing subject {subject}, file {i}/{len(subjects)} ===')

		# get file that contains merged episodes data for subject
		f = os.path.join(merged_episodes_folder, f'{subject}.csv')

		# read file as dataframe
		episodes = pd.read_csv(f)

		# read actigraph raw data for subject
		actigraph_acc, *_ = get_actigraph_acc_data(subject, hdf5_file = hdf5_read_file)

		# create new nw_vector
		nw_vector = np.zeros((actigraph_acc.shape[0], 1)).astype('uint8')
		
		# loop over each episode, extend the edges, and then record the non-wear time
		for _, row in episodes.iterrows():

			# only continue if the episode is non-wear time ( in the csv file , non wear time is encoded as 0. Note that we will flip this encoding in later stages so as to encode nw-time as 1)
			if row.loc['label'] == 0:
				# extract start and stop index
				start_index = row.loc['start_index']
				stop_index = row.loc['stop_index']
			
				# forward search to extend stop index
				stop_index = _forward_search_episode(actigraph_acc, stop_index, hz = 100, max_search_min = 5, std_threshold = std_threshold, verbose = False)
				# backwar search to extend start index
				start_index = _backward_search_episode(actigraph_acc, start_index, hz = 100, max_search_min = 5, std_threshold = std_threshold, verbose = False)
		
				# now update the non-wear vector
				nw_vector[start_index:stop_index] = 1
		
		# save non-wear vector to  to HDF5 file
		save_data_to_group_hdf5(group = 'true_nw_time', data = nw_vector, data_name = subject, overwrite = True, hdf5_file = hdf5_save_file)

def process_plot_merged_episodes(episodes_folder, grouped_episodes_folder, hdf5_file):
	"""
	Plot original non-merged episodes with merged episodes to see if the merging went ok.

	Parameters
	-----------
	episodes_folder : os.path()
		folder location of start and stop labels for non-wear time
	grouped_episodes_folder : os.path()
		folder location with episodes that have been merged when to appear close to each other
	hdf5_file : os.path
		location of HDF5 file that contains the raw activity data for actigraph and actiwave
	"""

	# read all files to process
	F = [f for f in read_directory(episodes_folder) if f[-4:] == '.csv']

	# process each file
	for f_idx, f in enumerate(F):

		# extract subject from file
		subject = re.search('[0-9]{8}', f)[0]

		logging.info(f'=== Processing subject {subject}, file {f_idx}/{len(F)} ===')

		# read csv file as dataframe
		episodes = pd.read_csv(f, index_col = 0)

		# read grouped episodes
		grouped_episodes = pd.read_csv(os.path.join(grouped_episodes_folder, f'{subject}.csv'), index_col = 0)

		"""
			READ ACCELERATION DATA
		"""

		# actigraph acceleration data
		actigraph_acc, _, actigraph_time = get_actigraph_acc_data(subject = subject, hdf5_file = hdf5_file)
		
		# actiwave acceleration data
		actiwave_acc, _, actiwave_time = get_actiwave_acc_data(subject = subject, hdf5_file = hdf5_file)
		
		# create dataframe
		df_actigraph_acc = pd.DataFrame(actigraph_acc, index = actigraph_time, columns = ['Y', 'X', 'Z'])
		df_actiwave_acc = pd.DataFrame(actiwave_acc, index = actiwave_time, columns = ['Y', 'X', 'Z'])

		# plot merged episode
		plot_merged_episodes(df_actigraph_acc, df_actiwave_acc, episodes, grouped_episodes, subject)


"""
	GRID SEARCH CNN
"""
def perform_grid_search_1d_cnn(label_folder, hdf5_read_file, save_data_location, save_model_folder, file_limit = None):
	"""
	Create a convolutional neural network through grid search. Here we explore the following grid search variables
		- episode window (how big a start or a stop episode needs to be)
		- cnn type (see v1, v2 etc in dl_functions), here we try out different architectures
	"""


	# read all files to process
	F = [f for f in read_directory(label_folder) if f[-4:] == '.csv'][:file_limit]

	"""
		GRID SEARCH VARIABLES
	"""
	# episode window in seconds
	EW = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	# cnn types
	CNN = ['v1', 'v2', 'v3', 'v4']
	
	"""
		DEEP LEARNING SETTINGS
	"""

	# define number of epochs
	epoch = 50
	# training proportion of the data
	train_split = 0.6
	# development proportion of the data (test size will be the remainder of train + dev)
	dev_split = 0.2
	
	"""
		OTHER SETTNGS
	"""
	save_features = True
	# load features from disk, can only be done if created first
	load_features_from_disk = False

	# perform combinations
	for idx, episode_window_sec in enumerate(EW):

		"""
			CREATE START AND STOP EPISODES
		"""

		try:

			# verbose
			logging.info(f'Processing combination {idx}/{len(EW)}')

			# verbose
			logging.debug(f'Episode window sec : {str(episode_window_sec)}')

			# calculate features if load_features_from_disk is set to False, otherwise read from disk (remember to have created them first)
			if not load_features_from_disk:
				
				# create the feature data
				executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
				# create tasks so we can execute them in parallel
				tasks = (delayed(get_start_and_stop_episodes)(file = f, label_folder = label_folder, hdf5_read_file = hdf5_read_file, idx = i, total = len(F), episode_window_sec = episode_window_sec) for i,  f in enumerate(F))
				
				# empty lists to hold x_0 and x_1
				new_data = {'x_0' : [], 'x_1' : []}

				# execute task
				for data in executor(tasks):
					# data contains x_0 and x_1 arrays
					for key, value in data.items():
						if len(value) > 0:
							new_data[key].append(value)

				# vstack all the arrays within new_data
				for key, value in new_data.items():
					new_data[key] = np.vstack(value)

				# define X_0 and X_1 as new variables and convert to float 32
				X_0 = np.array(new_data['x_0']).astype('float32')
				X_1 = np.array(new_data['x_1']).astype('float32')

				# upscale X_1
				X_1 = np.repeat(X_1, len(X_0) // len(X_1), 0)

				# create the Y features
				Y_0 = np.zeros((X_0.shape[0], 1))
				Y_1 = np.ones((X_1.shape[0], 1))
				
				# now stack features
				X = np.vstack([X_0, X_1])
				Y = np.vstack([Y_0, Y_1])

				# shuffle the dataset (necessary before we split into train, dev, test)
				X, Y = shuffle(X, Y, random_state = 42)

				if save_features:
					# construct save location
					save_features_location = os.path.join(save_data_location, str(episode_window_sec))
					# create directory if not exists
					create_directory(save_features_location)
					# save data as npz file
					np.savez(os.path.join(save_features_location, 'data.npz'), x = X, y = Y)
			else:
				
				# load features from disk
				features = np.load(os.path.join(save_data_location, str(episode_window_sec), 'data.npz'))
				X = features['x']
				Y = features['y']


			logging.debug(f'X shape : {X.shape}')
			logging.debug(f'Y shape : {Y.shape}')

			"""
				CREATE CNN MODEL
			"""

			for cnn_type in CNN:

				logging.info(f'Processing cnn_type : {cnn_type}')

				# dynamically create model name
				model_name = f'cnn_{cnn_type}_{episode_window_sec}.h5'

				# create 1D cnn
				create_1d_cnn_non_wear_episodes(X, Y, save_model_folder, model_name, cnn_type,  epoch = epoch, train_split = train_split, dev_split = dev_split, return_model = False)


		except Exception as e:
			
			logging.error(f'Unable to create 1D CNN model with EP: {episode_window_sec}, error : {e}')
			

def get_start_and_stop_episodes(file, label_folder, hdf5_read_file, episode_window_sec, idx = 1, total = 1, hz = 100, save_location = os.path.join(os.sep, 'users', 'shaheensyed', 'hdf5', 'start_stop_data')):
	"""
	Get start and stop episodes from activity data and save to disk

	Parameters
	-----------
	file : string
		file location of csv file that contains episodes
	label_folder : os.path
		folder location of start and stop labels for non-wear time
	save_label_folder : os.path
		folder location where to save the labels to for each subject together with type, window, subject, label, index, and counter information
	hdf5_read_file : os.path
		location of HDF5 file that contains raw acceleration data per participant

	"""

	# extract subject from file
	subject = re.search('[0-9]{8}', file)[0]

	logging.info(f'=== Processing subject {subject}, file {idx}/{total} ===')
	# logging.info(f'episode window sec : {episode_window_sec}, feature_window_ms : {feature_window_ms}, feature step ms : {feature_step_ms}')

	# read csv file with episodes
	episodes = pd.read_csv(file)

	# read actigraph raw data for subject
	actigraph_acc, *_ = get_actigraph_acc_data(subject, hdf5_file = hdf5_read_file)

	# empty list that will hold all the data
	data = {'x_0' : [], 'x_1' : []}

	# loop over each label and get start and stop episode
	for _, row in episodes.iterrows():

		# parse out variables from dataframe row
		# start = row.loc['start']
		start_index = row.loc['start_index']
		# stop = row.loc['stop']
		stop_index = row.loc['stop_index']
		label = 1 - row.loc['label']
		# counter = row.loc['counter']

		# logging.info(f'Counter : {counter}')

		# logging.info(f'Start_index : {start_index}, Stop_index: {stop_index}')

		# forward search to extend stop index
		stop_index = _forward_search_episode(actigraph_acc, stop_index, hz = 100, max_search_min = 5, std_threshold = 0.004, verbose = False)
		# backwar search to extend start index
		start_index = _backward_search_episode(actigraph_acc, start_index, hz = 100, max_search_min = 5, std_threshold = 0.004, verbose = False)

		# logging.info(f'Start_index : {start_index}, Stop_index: {stop_index}')

		# get start episode
		start_episode = actigraph_acc[start_index - (episode_window_sec * hz) : start_index]
		# get stop episode
		stop_episode = actigraph_acc[stop_index : stop_index + (episode_window_sec * hz)]
		
		# check if episode is right size
		if start_episode.shape[0] == episode_window_sec * hz:
			data[f'x_{label}'].append(start_episode)
		if stop_episode.shape[0] == episode_window_sec * hz:
			data[f'x_{label}'].append(stop_episode)

	return data


"""
	CREATE NON-WEAR TIME BY USING CNN MODEL
"""

def batch_process_get_nw_time_from_raw(hdf5_acc_file, hdf5_nw_file, limit = None, skip_n = 0):
	"""
	Batch process finding non-wear time in actigraph data based on mapping with actiwave data

	Parameters
	-----------
	limit : int (optional)
		limit the number of subjects to be processed
	skip_n : int (optional)
		skip first N subjects
	use_parallel = Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs = int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	"""

	# which CNN model to use
	model_folder = os.path.join('files', 'models', 'nw', 'cnn1d_60_20_20')
	cnn_type = 'v2'
	episode_window_sec = 6

	# grid parameters
	std_range = [0.004]
	edge_true_or_false_range = [True, False]
	start_stop_label_decision_range = ['or', 'and']
	distance_in_min_range = [1, 2, 3, 4, 5]

	for std in std_range:

		for edge_true_or_false in edge_true_or_false_range:

			for s_s in start_stop_label_decision_range:

				for d_in_min in distance_in_min_range:

					# settings
					hdf5_read_file = hdf5_acc_file
					hdf5_save_file = hdf5_nw_file
					std_threshold = std
					hz = 100
					# true non-wear time when one start/stop is classified as non-wear time, then use 'or', or both sides need to be classified as non-wear time, then use 'and'
					start_stop_label_decision = s_s
					# the distance in minutes when to group two candidate non-wear time episodes that are close to each other, how close is defined here in minutes
					distance_in_min = d_in_min

					# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
					subjects = get_all_subjects_hdf5(hdf5_file = hdf5_read_file)[0 + skip_n:limit]

					# exclude subjects that have issues with their data
					subjects = [s for s in subjects if s not in get_subjects_with_invalid_data()]

					# use parallel processing to speed up processing time
					executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
					# create tasks so we can execute them in parallel
					tasks = (delayed(process_get_nw_time_from_raw)(subject, model_folder, hdf5_read_file, hdf5_save_file, cnn_type, std_threshold, episode_window_sec, start_stop_label_decision, distance_in_min, edge_true_or_false, hz, i, len(subjects)) for i, subject in enumerate(subjects))
					# execute task
					executor(tasks)

def	process_get_nw_time_from_raw(subject, model_folder, hdf5_read_file, hdf5_save_file, cnn_type, std_threshold, episode_window_sec, start_stop_label_decision, distance_in_min, edge_true_or_false, hz, idx, total):
	"""
	Get non-wear time from raw acceleration data by using a trained cnn model

	
	Parameters
	-----------

	Returns
	-----------

	"""

	# verbose
	logging.info('{style} Processing subject: {} {}/{} {style}'.format(subject, idx, total, style = '='*10))

	# load cnn model
	cnn_model = load_tf_model(model_location = os.path.join(model_folder, cnn_type, f'cnn_{cnn_type}_{episode_window_sec}.h5'))

	# read actigraph raw data for subject
	actigraph_acc, _, actigraph_time = get_actigraph_acc_data(subject, hdf5_file = hdf5_read_file)

	# create new nw_vector
	nw_vector = np.zeros((actigraph_acc.shape[0], 1)).astype('uint8')

	"""
		FIND CANDIDATE NON-WEAR SEGMENTS ACTIGRAPH ACCELERATION DATA
	"""

	# get candidate non-wear episodes (note that these are on a minute resolution)
	nw_episodes = find_candidate_non_wear_segments_from_raw(actigraph_acc, std_threshold = std_threshold, min_segment_length = 1, sliding_window = 1, hz = hz)

	# flip the candidate episodes, we want non-wear time to be encoded as 1, and wear time encoded as 0
	nw_episodes = 1 - nw_episodes

	"""
		GET START AND END TIME OF NON WEAR SEGMENTS
	"""

	# find all indexes of the numpy array that have been labeled non-wear time
	nw_indexes = np.where(nw_episodes == 1)[0]
	# find consecutive ranges
	non_wear_segments = find_consecutive_index_ranges(nw_indexes)
	# empty dictionary where we can store the start and stop times
	dic_segments = {}

	# check if segments are found
	if len(non_wear_segments[0]) > 0:
		
		# find start and stop times (the indexes of the edges and find corresponding time)
		for i, row in enumerate(non_wear_segments):

			# find start and stop
			start, stop = np.min(row), np.max(row)

			# add the start and stop times to the dictionary
			dic_segments[i] = {'counter' : i, 'start': actigraph_time[start], 'stop' : actigraph_time[stop], 'start_index': start, 'stop_index' : stop}
	
	# create dataframe from segments
	episodes = pd.DataFrame.from_dict(dic_segments)


	"""
		MERGE EPISODES THAT ARE CLOSE TO EACH OTHER
	"""
				
	grouped_episodes = group_episodes(episodes = episodes.T, distance_in_min = distance_in_min, correction = 3, hz = hz, training = False).T

	"""
		FOR EACH EPISODE, EXTEND THE EDGES, CREATE FEATURES, AND INFER LABEL
	"""
	for _, row in grouped_episodes.iterrows():

		start_index = row.loc['start_index']
		stop_index = row.loc['stop_index']
	
		logging.info(f'Start_index : {start_index}, Stop_index: {stop_index}')

		# forward search to extend stop index
		stop_index = _forward_search_episode(actigraph_acc, stop_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = False)
		# backwar search to extend start index
		start_index = _backward_search_episode(actigraph_acc, start_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = False)

		logging.info(f'Start_index : {start_index}, Stop_index: {stop_index}')

		# get start episode
		start_episode = actigraph_acc[start_index - (episode_window_sec * hz) : start_index]
		# get stop episode
		stop_episode = actigraph_acc[stop_index : stop_index + (episode_window_sec * hz)]

		# label for start and stop combined
		start_stop_label = [False, False]

		"""
			START EPISODE
		""" 
		if start_episode.shape[0] == episode_window_sec * hz:

			# reshape into num feature x time x axes
			start_episode = start_episode.reshape(1, start_episode.shape[0], start_episode.shape[1]) 
			
			# get binary class from model
			start_label = cnn_model.predict_classes(start_episode).squeeze()

			# if the start label is 1, this means that it is wear time, and we set the first start_stop_label to 1
			if start_label == 1:
				start_stop_label[0] = True	
		
		else:
			# there is an episode right at the start of the data, since we cannot obtain a full epsisode_window_sec array
			# here we say that True for nw-time and False for wear time
			start_stop_label[0] = edge_true_or_false
		

		"""
			STOP EPISODE
		""" 
		if stop_episode.shape[0] == episode_window_sec * hz:
			
			# reshape into num feature x time x axes
			stop_episode = stop_episode.reshape(1, stop_episode.shape[0], stop_episode.shape[1]) 
			
			# get binary class from model
			stop_label = cnn_model.predict_classes(stop_episode).squeeze()

			# if the start label is 1, this means that it is wear time, and we set the first start_stop_label to 1
			if stop_label == 1:
				start_stop_label[1] = True	
		else:
			# there is an episode right at the END of the data, since we cannot obtain a full epsisode_window_sec array
			# here we say that True for nw-time and False for wear time
			start_stop_label[1] = edge_true_or_false
		
		if start_stop_label_decision == 'or':
			# use logical OR to determine if episode is true non-wear time
			if any(start_stop_label):
				# true non-wear time, record start and stop in nw-vector
				nw_vector[start_index:stop_index] = 1
		elif start_stop_label_decision == 'and':
			# use logical and to determine if episode is true non-wear time
			if all(start_stop_label):
				# true non-wear time, record start and stop in nw-vector
				nw_vector[start_index:stop_index] = 1
		else:
			logging.error(f'Start/Stop decision unknown, can only use or/and, given: {start_stop_label_decision}')

	# save nw_vector to HDF5
	group_name = f'{cnn_type}_{episode_window_sec}_{start_stop_label_decision}_{distance_in_min}_{std_threshold}_{edge_true_or_false}'

	# save to HDF5 file
	save_data_to_group_hdf5(group = group_name, data = nw_vector, data_name = subject, overwrite = True, hdf5_file = hdf5_save_file)

def calculate_cnn_classification_performance(hdf5_read_file):

	
	# which cnn type to use
	cnn_type = 'v2'
	# window length in seconds
	episode_window_sec = 6

	# grid parameters
	std_threshold_range = [0.004]
	edge_true_or_false_range = [True, False]
	start_stop_label_decision_range = ['or', 'and']
	distance_in_min_range = [2, 3, 4]
	distance_in_min_range = [1,5]

	for std_threshold in std_threshold_range:

		for edge_true_or_false in edge_true_or_false_range:

			for start_stop_label_decision in start_stop_label_decision_range:

				for distance_in_min in distance_in_min_range:

					# verbose
					logging.info(f'Processing CNN_TYPE : {cnn_type}, EPISODE WINDOW SEC : {episode_window_sec}, STD RANGE : {std_threshold}, EDGE : {edge_true_or_false}, START/STOP LABEL : {start_stop_label_decision}, DISTANCE : {distance_in_min}')
					
					# empty dictionary to hold all the data
					data = {}

					"""
						TRUE NON WEAR TIME
					"""

					logging.info('Reading true non-wear time')

					# read data of true non-wear time
					for subject in get_datasets_from_group(group_name = 'true_nw_time', hdf5_file = hdf5_read_file):

						# read dataset
						nw_time = read_dataset_from_group(group_name = 'true_nw_time', dataset = subject, hdf5_file = hdf5_read_file)

						data[subject] = {'y' : nw_time, 'y_hat' : None}
						
					"""
						INFERRED NON WEAR TIME
					"""

					# group name of inferred non-wear time
					inferred_nw_time_group_name = f'{cnn_type}_{episode_window_sec}_{start_stop_label_decision}_{distance_in_min}_{std_threshold}_{edge_true_or_false}'

					logging.info('Reading inferred non-wear time')

					# read data of true non-wear time
					for subject in data.keys():

						# read dataset
						inferred_nw_time = read_dataset_from_group(group_name = inferred_nw_time_group_name, dataset = subject, hdf5_file = hdf5_read_file)

						data[subject]['y_hat'] = inferred_nw_time
					
					"""
						CALCULATE tn, fp, fn, tp
					"""
					# empty dataframe to hold values
					all_results = pd.DataFrame()

					logging.info('Calculating classification performance')

					# create the feature data
					executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
					# create tasks so we can execute them in parallel
					tasks = (delayed(_calculate_confusion_matrix)(key, value) for key,  value in data.items())
					
					# execute task
					for key, series in executor(tasks):
						all_results[key] = pd.Series(series)
					
					# tranpose dataframe
					all_results = all_results.T

					# save to CSV
					all_results.to_csv(os.path.join('files', 'cnn_nw_performance', f'{inferred_nw_time_group_name}_per_subject.csv'))
					
					
					tn = all_results['tn'].sum()
					fp = all_results['fp'].sum()
					fn = all_results['fn'].sum()
					tp = all_results['tp'].sum()

					logging.debug('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

					# calculate classification performance such as precision, recall, f1 etc.
					classification_performance = calculate_classification_performance(tn, fp, fn, tp)

					df_classification_performance = pd.DataFrame(pd.Series(classification_performance))
					df_classification_performance.to_csv(os.path.join('files', 'cnn_nw_performance', f'{inferred_nw_time_group_name}_all_subjects.csv'))


"""
	EVALUATE BASELINE MODELS
"""

def batch_evaluate_baseline_models(hdf5_acc_file, hdf5_nw_file):

	# standard deviation range
	std_threshold_range = [0.004, 0.005, 0.006, 0.007]
	# episode length
	episode_length_range = [15, 30, 45, 60, 75, 90, 105, 120]
	# create combinations
	combinations = [ (std_threshold, episode_length) for std_threshold in std_threshold_range for episode_length in episode_length_range]

	# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
	subjects = get_all_subjects_hdf5(hdf5_file = hdf5_acc_file)

	# exclude subjects that have issues with their data
	subjects = [s for s in subjects if s not in get_subjects_with_invalid_data()]

	# create parallel executor
	executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
	# create tasks so we can execute them in parallel
	tasks = (delayed(_read_true_non_wear_time_from_hdf5)(subject, hdf5_nw_file, i, len(subjects)) for i, subject in enumerate(subjects))
	
	# empty dictionary to hold true non wear time
	true_nw_time = {}

	logging.info('Reading true non-wear time')

	# execute task and return data
	for subject, nw_time_vector in executor(tasks):
		# add true non wear time to dictionary
		true_nw_time[subject] = nw_time_vector

	# evaluate each combination
	for i, combination in enumerate(combinations):

		# unpack tuple variables
		std_threshold, episode_length = combination

		# verbose
		logging.info(f'Processing std_treshold : {std_threshold}, episode length : {episode_length} {i}/{len(combinations)}')

		# call evaluate function
		evaluate_baseline_models(subjects, hdf5_acc_file, true_nw_time, std_threshold, episode_length)


def evaluate_baseline_models(subjects, hdf5_acc_file, true_nw_time, std_threshold, episode_length, use_vmu = True, save_folder = os.path.join('files', 'baseline_performance_vmu')):

	# create the feature data
	executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')

	# create tasks so we can execute them in parallel
	tasks = (delayed(_evaluate_baseline)(hdf5_acc_file, subject, std_threshold, episode_length, use_vmu, i, len(subjects)) for i, subject in enumerate(subjects))
	
	# new dictionary to keep all data
	data = {}

	# execute task and return data
	for subject, y_hat in executor(tasks):

		y = true_nw_time[subject]
		# keep y and y_hat in dictionary
		data[subject] = {'y' : y, 'y_hat' : y_hat}

	"""
		CALCULATE tn, fp, fn, tp
	"""

	# empty dataframe to hold values
	all_results = pd.DataFrame()

	logging.info('Calculating classification performance')

	# create the feature data
	executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
	# create tasks so we can execute them in parallel
	tasks = (delayed(_calculate_confusion_matrix)(key, value) for key,  value in data.items())
	
	# execute task
	for key, series in executor(tasks):
		all_results[key] = pd.Series(series)
	
	# transpose dataframe
	all_results = all_results.T

	# create save folder if not exists
	create_directory(save_folder)

	# save to CSV
	all_results.to_csv(os.path.join(save_folder, f'{std_threshold}_{episode_length}_per_subject.csv'))
	
	
	tn = all_results['tn'].sum()
	fp = all_results['fp'].sum()
	fn = all_results['fn'].sum()
	tp = all_results['tp'].sum()

	logging.debug('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

	# calculate classification performance such as precision, recall, f1 etc.
	classification_performance = calculate_classification_performance(tn, fp, fn, tp)

	df_classification_performance = pd.DataFrame(pd.Series(classification_performance))
	df_classification_performance.to_csv(os.path.join(save_folder, f'{std_threshold}_{episode_length}_all.csv'))
				
"""
	PLOT CLASSIFICATION PERFORMANCE
"""

def process_plot_cnn_classification_performance(model_folder):

	# all data for plotting
	cnn_types = ['v1', 'v2', 'v3', 'v4']

	plot_data = {x : None for x in cnn_types}

	episode_window_sec = [2, 3, 4, 5, 6, 7, 8, 9]

	# epoch data to use
	epoch = 50

	# columns to keep and rename to
	columns = { 'accuracy' : 'accuracy train',
				'precision' : 'precision train',
				'recall' : 'recall train',
				'F1 train' : 'F1 train',
				#'auc' : 'AUC train',
				'val_accuracy' : 'accuracy val',
				'val_precision' : 'precision val',
				'val_recall' : 'recall val',
				'F1 val' : 'F1 val',
				#'val_auc' : 'AUC val',
				'test_accuracy' : 'accuracy test',
				'test_precision' : 'precision test',
				'test_recall' : 'recall test',
				'F1 test' : 'F1 test',
				#'test_auc' : 'AUC test',
	}

	for cnn_type in cnn_types:

		data = pd.DataFrame()

		for ew in episode_window_sec:

			# dynamically create training history file name based on cnn_type and episode window
			training_history = os.path.join(model_folder, cnn_type, f'cnn_{cnn_type}_{ew}.h5_history.csv')
			test_history = os.path.join(model_folder, cnn_type, f'cnn_{cnn_type}_{ew}.h5_history_test.csv')

			# load data as dataframe
			df_training = pd.read_csv(training_history)
			df_test = pd.read_csv(test_history, index_col = 0)

			# add epoch row to all data dataframe
			data[ew] = pd.concat([df_training.iloc[epoch-1], df_test.T.iloc[0]], axis = 0)
		
		# tranpose dataframe
		data = data.T

		# calculate training, dev, and test F1 score
		data['F1 train'] = 2 * data['tp'] / (2 * data['tp'] + data['fp'] + data['fn'])
		data['F1 val'] = 2 * data['val_tp'] / (2 * data['val_tp'] + data['val_fp'] + data['val_fn'])
		data['F1 test'] = 2 * data['test_tp'] / (2 * data['test_tp'] + data['test_fp'] + data['test_fn'])

		data = data[[x for x in columns.keys()]]
		data = data.rename(columns = columns)

		plot_data[cnn_type] = data.T


	plot_cnn_classification_performance(plot_data)

def process_plot_cnn_training_results_per_epoch(model_folder, episode_window, cnn_type):
	"""
	Plot training results from model history
	"""

	# history file 
	history_file = os.path.join(model_folder, cnn_type, f'cnn_{cnn_type}_{episode_window}.h5_history.csv')

	# load dataframe
	data = pd.read_csv(history_file, index_col = 0)

	# add F1 results
	data['F1'] = 2 * data['tp'] / (2 * data['tp'] + data['fp'] + data['fn'])
	data['val_F1'] = 2 * data['val_tp'] / (2 * data['val_tp'] + data['val_fp'] + data['val_fn'])

	# call plot function
	plot_training_results_per_epoch(data)

def process_plot_cnn_inferred_nw_time(hdf5_acc_file, hdf5_nw_file):

	# v2_6_and_4_0.004_True_all_subjects
	cnn_type = 'v2'
	episode_window_sec = 6
	start_stop_label_decision = 'and'
	distance_in_min = 4
	std_threshold = 0.004
	edge_true_or_false = True

	# group that contains data on inferred non-wear time
	# inferred_nw_group = f'{cnn_type}_{episode_window_sec}_{start_stop_label_decision}_{distance_in_min}_{std_threshold}'
	inferred_nw_group = f'{cnn_type}_{episode_window_sec}_{start_stop_label_decision}_{distance_in_min}_{std_threshold}_{edge_true_or_false}'

	# subjects = ['90042823', '90086831','90173929','90233320','90301821','90341522','90446124','90486330','90489737','90902626','91008321','91845330',
	# 			'92198837','92723932','92987439','90021214','90015722','90089632','90110920','90172019','90181524','90232925','90239124','90242017',
	# 			'90287430','90319123','90345122','90359733','90389837','90413825','90423321','90537529','90681428','90923225','90925328','91643427',
	# 			'92150421','92191527','93214322','93324223']

	# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
	subjects = get_all_subjects_hdf5(hdf5_file = hdf5_acc_file)

	# exclude subjects that have issues with their data
	subjects = [s for s in subjects if s not in get_subjects_with_invalid_data()]


	for subject in subjects:

		# read inferred non-wear time
		inferred_nw_time = read_dataset_from_group(group_name = inferred_nw_group, dataset = subject, hdf5_file = hdf5_nw_file)

		# read true non_wear time
		true_nw_time = read_dataset_from_group(group_name = 'true_nw_time', dataset = subject, hdf5_file = hdf5_nw_file)
		
		# read actigraph raw data for subject
		actigraph_acc, _, actigraph_time = get_actigraph_acc_data(subject, hdf5_file = hdf5_acc_file)

		# combine all data into dataframe
		data = pd.DataFrame(actigraph_acc, index = actigraph_time, columns = ['Y', 'X', 'Z'])

		# add infered non wear time
		data['INFERRED NW-TIME'] = inferred_nw_time
		# add true non-wear time
		data['TRUE NW-TIME'] = true_nw_time

		plot_cnn_inferred_nw_time(subject, data)

def process_plot_episodes_used_for_training(merged_episodes_folder, hdf5_acc_file, hz = 100, std_threshold = 0.004):

	# read csv files in merged episodes folder
	F = read_directory(merged_episodes_folder)

	# list to hold plot data
	plot_data = {'0' : [], '1' : []}

	# define the length of the window in seconds
	episode_window_sec = 20 * hz
	# define how much of the flat line nees to be shown in the plot
	show_flat_sec = 20 * hz

	# loop over each file in F
	for f in F[200:]:

		if len(plot_data['0']) > 20 and len(plot_data['1']) > 20:
			break

		# extract subject from file
		subject = re.search('[0-9]{8}', f)[0]

		logging.info(f'=== Processing subject {subject} ===')

		# read actigraph raw data for subject
		actigraph_acc, *_ = get_actigraph_acc_data(subject, hdf5_file = hdf5_acc_file)

		# read dataframe
		df = pd.read_csv(f)

		for _, row in df.iterrows():

			start_index = row.loc['start_index']
			stop_index = row.loc['stop_index']
			label = row.loc['label']
			
			# forward search to extend stop index
			stop_index = _forward_search_episode(actigraph_acc, stop_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = False)
			# backwar search to extend start index
			start_index = _backward_search_episode(actigraph_acc, start_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = False)

			# get start episode
			start_episode = actigraph_acc[start_index - episode_window_sec : start_index + show_flat_sec]
			# get stop episode
			stop_episode = actigraph_acc[stop_index - show_flat_sec: stop_index + episode_window_sec]
			
			# check if start_episode has the right shape
			if start_episode.shape[0] == episode_window_sec + show_flat_sec:
				if np.all(np.std(start_episode, axis = 0) > 0.1):
					plot_data[f'{label}'].append((subject, start_episode))				
			if stop_episode.shape[0] == episode_window_sec + show_flat_sec:
				if np.all(np.std(stop_episode, axis = 0) > 0.1):
					plot_data[f'{label}'].append((subject, stop_episode))

	# plot the data
	plot_episodes_used_for_training_combined(plot_data)
	# plot_episodes_used_for_training(plot_data)

def process_plot_baseline_performance(data_folder):

	"""
	Plot classification performance of baseline models
	"""

	# subfolders
	subfolders = ['VMU', 'NO_VMU']

	for subfolder in subfolders:

		performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
		# standard deviation range
		std_threshold_range = [0.003, 0.004, 0.005, 0.006, 0.007]
		# episode length
		episode_length_range = [15, 30, 45, 60, 75, 90, 105, 120]
		# create dictionary with metrics and dataframe to populate data to
		plot_data = {x : pd.DataFrame(columns = episode_length_range, index = std_threshold_range) for x in performance_metrics}

		# get all files that contain classification performance of all subjects
		for f in [x for x in read_directory(os.path.join(data_folder, subfolder)) if 'all' in x]:

			# extract std threshold from fil ename
			std_threshold = float(f.split(os.sep)[-1].split('_')[0])
			# extract episode length from file name
			episode_length = int(f.split(os.sep)[-1].split('_')[1])

			if std_threshold in std_threshold_range:
				if episode_length in episode_length_range:
					# load file as dataframe, and take first columns, which makes it a seriers
					f_data = pd.read_csv(f, index_col = 0)['0']

					# loop values in f_data
					for key, value in f_data.items():
						# only look at keys that are part of the performance metrics list
						if key in performance_metrics:
							# populate correct dataframe (based on row, column combination)
							plot_data[key].at[std_threshold, episode_length] = value
		
		plot_baseline_performance(plot_data, plot_name = subfolder)

def process_plot_performance_cnn_nw_method(data_folder):
	"""
	Plot the performance of the nw time algortihm that uses the cnn classification model
	"""

	performance_metrics = ['accuracy', 'precision', 'recall', 'f1']

	# read files in folder
	F = [x for x in read_directory(data_folder) if 'all' in x]

	# empty dataframe
	df = pd.DataFrame()#index = performance_metrics)
	
	for f in F:

		# extract variables from file name
		cnn_type, episode_window_sec, start_stop_label_decision, distance_in_min, std_threshold, edge_true_or_false, *_ = f.split(os.sep)[-1].split('_')
		
		# read data as series
		f_data = pd.read_csv(f, index_col = 0)['0']
		f_data['cnn_type'] = cnn_type
		f_data['episode_window_sec'] = episode_window_sec
		f_data['start_stop_label_decision'] = start_stop_label_decision
		f_data['distance_in_min'] = distance_in_min
		f_data['std_threshold'] = std_threshold
		f_data['edge_true_or_false'] = edge_true_or_false

		edge_true_or_false = 'nw time' if eval(edge_true_or_false) else 'wear time'

		df[f'{start_stop_label_decision.upper()}, {distance_in_min} min, {edge_true_or_false}'] = f_data
	
	# transpose dataframe
	df = df.T.sort_index()

	print(df)
	# # call plot function
	# plot_performance_cnn_nw_method(df)
	

"""
	INTERNAL HELPER FUNCTION
"""

def _filter_true_start_stop_indexes_from_candidates(indexes, candidate_type, candidate_indexes, true_nw_indexes, range_window = 10):
	"""
	There are two lists of array. (1) with candidate non wear segments indexes for start|stop episode, and (2) indexes for true non-wear time for start|stop
	The problem is that the true indexes might not be within the candidate indexes exactly, but the value might be slightly off. For instance, a true index could be 99
	but the candidate has the value 100. Then this 100 is considered to be true and that is what we're trying to filter out here.

	Paramaters
	----------
	indexes : dic()
		dictionary that holds labels and indexes
	candidate_type : string
		either 'start' or 'stop to indicate a starting episode or a stopping episode
	candidate_indexes : np.array
		list of indexes of candidate non wear episodes of the type 'start' or 'stop'
	true_nw_indexes : np.array
		list of indexes that indicate the start or stop of a true non wear episode
	range_window : int
		true indexes but not exactly match the values found within candidate indexes but they are typically off by a few, this numbers is used to indidcate by how much they could be off.
	
	Returns
	--------
	indexes : dic()
		dictionary that holds labels and indexes but now updated with the results of this function
	"""

	# empty list to hold all true indexes (including the range)
	true_indexes = []
	# get all true indexes including range
	for true_idx in true_nw_indexes[0]:

		# increase index range
		for true_range_idx in true_idx - (range_window // 2) + range(range_window):
			true_indexes.append(true_range_idx)
	
	# loop over each candidate index
	for can_idx in candidate_indexes[0]:

			# if candidate index part of true index range, then this is considered true non wear time, otherwise, its not	
			if can_idx in true_indexes:		
				indexes[f'true_{candidate_type}']['indexes'].append(can_idx)
			else:
				indexes[f'false_{candidate_type}']['indexes'].append(can_idx)

	return indexes

def _forward_search_episode(acc_data, index, hz = 100, max_search_min = 5, std_threshold = 0.004, verbose = False):
	"""
	When we have an episode, this was created on a minute resolution, here we do a forward search to find the edges of the episode with a second resolution
	"""

	# calculate max slice index
	max_slice_index = acc_data.shape[0]

	for i in range(hz * 60 * max_search_min):

		# create new slices
		new_start_slice = index
		new_stop_slice = index + hz

		if verbose:
			logging.info(f'i : {i}, new_start_slice : {new_start_slice}, new_stop_slice : {new_stop_slice}')

		# check if the new stop slice exceeds the max_slice_index
		if new_stop_slice > max_slice_index:
			if verbose:
				logging.info(f'Max slice index reached : {max_slice_index}')
			break
			
		# slice out new activity data
		slice_data = acc_data[new_start_slice:new_stop_slice]

		# calculate the standard deviation of each column (YXZ)
		std = np.std(slice_data, axis=0)
		
		# check if all of the standard deviations are below the standard deviation threshold
		if np.all(std <= std_threshold):
			
			# update index
			index = new_stop_slice
		else:
			break

	if verbose:
		logging.info(f'New index : {index}, number of loops : {i}')
	return index

def _backward_search_episode(acc_data, index, hz = 100, max_search_min = 5, std_threshold = 0.004, verbose = False):
	"""
	When we have an episode, this was created on a minute resolution, here we do a backward search to find the edges of the episode with a second resolution
	"""


	# calculate min slice index
	min_slice_index = 0

	for i in range(hz * 60 * max_search_min):

		# create new slices
		new_start_slice = index - hz
		new_stop_slice = index

		if verbose:
			logging.info(f'i : {i}, new_start_slice : {new_start_slice}, new_stop_slice : {new_stop_slice}')

		# check if the new start slice exceeds the max_slice_index
		if new_start_slice < min_slice_index:
			if verbose:
				logging.debug(f'Minimum slice index reached : {min_slice_index}')
			break
			
		# slice out new activity data
		slice_data = acc_data[new_start_slice:new_stop_slice]

		# calculate the standard deviation of each column (YXZ)
		std = np.std(slice_data, axis=0)
		
		# check if all of the standard deviations are below the standard deviation threshold
		if np.all(std <= std_threshold):
			
			# update index
			index = new_start_slice
		else:
			break

	if verbose:
		logging.info(f'New index : {index}, number of loops : {i}')
	return index

def _calculate_confusion_matrix(key, values, hz = 100):

	# unpack y and y_hat
	y = values['y']
	y_hat = values['y_hat']

	# downscale to second level
	y = y[::hz]
	y_hat = y_hat[::hz]

	# get confusion matrix values
	tn, fp, fn, tp = get_confusion_matrix(y, y_hat, labels = [0,1]).ravel()

	logging.debug('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

	return key, {'tn': tn, 'fp' : fp, 'fn' : fn, 'tp' : tp}

def _read_true_non_wear_time_from_hdf5(subject, hdf5_nw_file, idx = 1, total = 1):

	"""
		READ TRUE NON-WEAR TIME
	"""
	logging.debug(f'\rProcessing subject : {subject} {idx}/{total}')

	true_nw_time = read_dataset_from_group(group_name = 'true_nw_time', dataset = subject, hdf5_file = hdf5_nw_file)

	return subject, true_nw_time

def _evaluate_baseline(hdf5_acc_file, subject, std_threshold, episode_length, use_vmu, i = 1, total = 1, hz = 100):

	# verbose
	logging.debug(f'Processing subject {i}/{total}')
	"""
		READ SUBJECT DATA
	"""

	# read actigraph raw data for subject
	actigraph_acc, *_ = get_actigraph_acc_data(subject, hdf5_file = hdf5_acc_file)

	# create new nw_vector
	nw_vector = np.zeros((actigraph_acc.shape[0], 1)).astype('uint8')

	"""
		FIND CANDIDATE NON-WEAR SEGMENTS ACTIGRAPH ACCELERATION DATA
	"""

	# get candidate non-wear episodes (note that these are on a minute resolution)
	nw_episodes = find_candidate_non_wear_segments_from_raw(actigraph_acc, std_threshold = std_threshold, min_segment_length = 1, sliding_window = 1, hz = hz, use_vmu = use_vmu)

	# flip the candidate episodes, we want non-wear time to be encoded as 1, and wear time encoded as 0
	nw_episodes = 1 - nw_episodes

	"""
		GET START AND END TIME OF NON WEAR SEGMENTS
	"""

	# find all indexes of the numpy array that have been labeled non-wear time
	nw_indexes = np.where(nw_episodes == 1)[0]
	# find consecutive ranges
	non_wear_segments = find_consecutive_index_ranges(nw_indexes)

	# check if segments are found
	if len(non_wear_segments[0]) > 0:
		
		# find start and stop times (the indexes of the edges and find corresponding time)
		for i, row in enumerate(non_wear_segments):

			# find start and stop
			start, stop = np.min(row), np.max(row)

			# calculate lenght of episode in minutes
			length = int((stop - start) / hz / 60)

			# check if length exceeds threshold, if so, then this is non-wear time
			if length >= episode_length:
				# now update nw vector
				nw_vector[start:stop] = 1

	# return values
	return subject, nw_vector

# start code here
if __name__ == '__main__':

	# start timer and memory counter
	tic, process, logging = set_start()

	# define environement
	env = 'local' if os.uname().nodename == 'shaheenmbp2' else 'server'
	
	# location of hdf5 files
	if env == 'local':

		# HDF5 file that contains raw acceleration data
		actiwave_actigraph_mapping_hdf5_file = os.path.join(os.sep, 'users', 'shaheensyed', 'hdf5', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')
		# HDF5 file that contains nw-time data or that will be used to save nw-time data to
		nw_time_hdf5_file = os.path.join(os.sep, 'users', 'shaheensyed', 'hdf5', 'NW_TIME.hdf5')
		# location with csv files that contain episodes with start, stop, start_index, stop_index, and label
		episodes_folder = os.path.join('labels', 'start_stop_all')
		# location where to save the merged episodes to
		merged_episodes_folder = os.path.join('labels', 'start_stop_all_grouped')
		# location with cnn nw time classification results
		cnn_nw_classification_folder = os.path.join('files', 'cnn_nw_performance')
	
	elif env == 'server':

		# HDF5 file that contains raw acceleration data
		actiwave_actigraph_mapping_hdf5_file = os.path.join(os.sep, 'home', 'shaheen', 'Documents', 'hdf5', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')
		# HDF5 file that contains nw-time data or that will be used to save nw-time data to
		nw_time_hdf5_file = os.path.join(os.sep, 'media', 'shaheen',  'LaCie_serve', 'NW_TIME.hdf5')
		# location with csv files that contain episodes with start, stop, start_index, stop_index, and label
		episodes_folder = os.path.join('labels', 'start_stop_all')
		# location where to save the merged episodes to
		merged_episodes_folder = os.path.join('labels', 'start_stop_all_grouped')
		# location with cnn nw time classification results
		cnn_nw_classification_folder = os.path.join('files', 'cnn_nw_performance')

	else:
		logging.error(f'Unknown environment variable : {env}')
		exit(1)

	"""
		1) PREPROCESSING
	"""

	# 1a) merge episodes that are close to each other
	# merge_close_episodes_from_training_data(read_folder = episodes_folder, save_folder = merged_episodes_folder)

	# 1b) calculate true non-wear time from labeled episodes
	# process_calculate_true_nw_time_from_labeled_episodes(merged_episodes_folder = merged_episodes_folder, hdf5_read_file = actiwave_actigraph_mapping_hdf5_file, hdf5_save_file = nw_time_hdf5_file )

	# 1c) plot episodes, both normal and grouped
	# process_plot_merged_episodes(episodes_folder = episodes_folder, grouped_episodes_folder = merged_episodes_folder, hdf5_file = actiwave_actigraph_mapping_hdf5_file)

	"""
		2) PERFORM GRID SEARCH FOR DIFFERENT SIZED FEATURE WINDOWS
	"""

	# 2) create cnn model with grid search
	# perform_grid_search_1d_cnn(	label_folder = merged_episodes_folder, \
	# 							hdf5_read_file = actiwave_actigraph_mapping_hdf5_file, \
	# 							save_data_location = os.path.join(os.sep, 'users', 'shaheensyed', 'hdf5', 'start_stop_data'),\
	# 							save_model_folder = os.path.join('files', 'models', 'nw', 'cnn1d'),\
	# 							file_limit = None)

	"""
		3) CREATE NON-WEAR TIME BY USING CNN MODEL
	"""
	# batch_process_get_nw_time_from_raw(hdf5_acc_file = actiwave_actigraph_mapping_hdf5_file, hdf5_nw_file = nw_time_hdf5_file)

	"""
		4) CALCULATE CLASSIFICATION PERFORMANCE CNN MODEL COMPARED TO TRUE NON-WEAR TIME
	"""
	# calculate_cnn_classification_performance(hdf5_read_file = nw_time_hdf5_file)

	"""
		5) EVALUATE BASELINE MODELS
	"""
	# batch_evaluate_baseline_models(hdf5_acc_file = actiwave_actigraph_mapping_hdf5_file, hdf5_nw_file = nw_time_hdf5_file)
	
	"""
		5) PAPER PLOTS
	"""
	# process_plot_cnn_classification_performance(model_folder = os.path.join('files', 'models', 'nw', 'cnn1d'))

	# process_plot_cnn_training_results_per_epoch(model_folder = os.path.join('files', 'models', 'nw', 'cnn1d'), cnn_type = 'v2', episode_window = 6)

	# process_plot_cnn_inferred_nw_time(hdf5_acc_file = actiwave_actigraph_mapping_hdf5_file, hdf5_nw_file = nw_time_hdf5_file)

	# plot wear and nw-time episodes used for training
	# process_plot_episodes_used_for_training(merged_episodes_folder, actiwave_actigraph_mapping_hdf5_file)

	# plot baseline performance
	# process_plot_baseline_performance(data_folder = os.path.join('files', 'baseline_performance'))

	# plot the performance of the new cnn nw-time algorithm
	# process_plot_performance_cnn_nw_method(data_folder = cnn_nw_classification_folder)

	set_end(tic,process)

	