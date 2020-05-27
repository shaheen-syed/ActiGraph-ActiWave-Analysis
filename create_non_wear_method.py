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
from functions.plot_functions import 	plot_merged_episodes, plot_cnn_classification_performance, plot_training_results_per_epoch,\
										plot_baseline_performance, plot_start_stop_segments,\
										plot_baseline_performance_compare_f1, plot_overview_all_raw_nw_methods
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

	Note that the words merge and group are used here interchangably

	

	Parameters
	-----------
	read_folder : os.path()
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
	"""
	Read labeled episodes and create a non-wear vector with gold standard labels.
	These labels have a start and stop timestamp with a 1 minute resolution. We also extend the edges to obtain a resolution on a 1-second level.
	This extension is done by incrementally extending the edges with 1-second intervals. If the intervals are below the std_threshold, then include that interval into 
	the non-wear episode.

	Paramaters
	-------------
	merged_episodes_folder : os.path
		folder location where episodes are stored that have undergone the merged function. See function merge_close_episodes_from_training_data 
	hdf5_read_file : os.path
		file location of the HDF5 file that contains all the raw data
	hdf5_save_file : os.path
		file name to create a new HDF5 file for
	std_threshold : float (optional)
		standard deviation threshold that is used to calculate if the acceleration is below this value. The 0.004 threshold is used to find episodes where the acceleration
		is flat (i.e., no activity)
	"""

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
	A total of four 1D CNN architectures were constructed and trained for the binary classification of our features as either belonging 
	to true non-wear time or to wear time; Figure 2 shows the four proposed architectures labelled V1, V2, V3, and V4. The input feature 
	is a vector of w x 3 (i.e. three orthogonal axes), where w is the window size ranging from 2–10 seconds. In total, 10 x 4 = 40 different 
	CNN models were trained. CNN V1 can be considered a basic CNN with only a single convolutional layer followed by a single fully connected 
	layer. CNN V2 and V3 contain additional convolutional layers with different kernel sizes and numbers of filters. Stacking convolutional 
	layers enables the detection of high-level features, unlike single convolutional layers. CNN V4 contains a max pooling layer after each 
	convolutional layer to merge semantically similar features while reducing the data dimensionality.(LeCun et al., 2015) A CNN architecture 
	with max pooling layers has shown varying results, from increased classification performance (Song-Mi Lee et al., 2017) to pooling layers 
	interfering with the convolutional layer’s ability to learn to down sample the raw sensor data.(Ordóñez & Roggen, 2016) All proposed CNN 
	architectures have a single neuron in the output layer with a sigmoid activation function for binary classification. 

	Training was performed on 60% of the data, with 20% used for validation and another 20% used for testing. All models were trained for up to 250 
	epochs with the Adam optimiser (Kingma & Ba, 2014) and a learning rate of 0.001. Loss was calculated with binary cross entropy and, additionally, 
	early stopping was implemented to monitor the validation loss with a patience of 25 epochs and restore weights of the lowest validation loss. 
	This means that training would terminate if the validation loss did not improve for 25 epochs, and the best model weights would be restored. All 
	models were trained on 2 x Nvidia RTX 2080TI graphics cards with the Python library TensorFlow (v2.0).

	Parameters
	----------
	label_folder : os.path
		folder location where gold standard labels are saved
	hdf5_read_file : os.path
		location of HDF5 file that contains the raw activity data for actigraph and actiwave
	save_data_location : os.path
		folder location where to save the training features to
	save_model_folder : os.path
		folder location where to save the trained CNN model to
	file_limit : int (optional)
		can be used for debugging/testing since it limits the number of features to be created
	"""

	# read all files to process
	F = [f for f in read_directory(label_folder) if f[-4:] == '.csv'][:file_limit]

	"""
		GRID SEARCH VARIABLES
	"""
	# episode window in seconds
	EW = [2, 3, 4, 5, 6, 7, 8, 9, 10]
	# cnn architectures (see dl_functions about the actual architecture of these types)
	CNN = ['v1', 'v2', 'v3', 'v4']
	
	"""
		DEEP LEARNING SETTINGS
	"""

	# define number of epochs
	epoch = 250
	# training proportion of the data
	train_split = 0.6
	# development proportion of the data (test size will be the remainder of train + dev)
	dev_split = 0.2
	
	"""
		OTHER SETTNGS
	"""
	# set to true if created features need to be stored to disk (this can then be used to read from disk which is much faster)
	save_features = False
	# load features from disk, can only be done if created first (see save_features Boolean value)
	load_features_from_disk = True

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
	Get start and stop episodes from activity data

	Parameters
	-----------
	file : string
		file location of csv file that contains episodes
	label_folder : os.path
		folder location of start and stop labels for non-wear time
	hdf5_read_file : os.path
		location of HDF5 file that contains raw acceleration data per participant
	episode_window_sec : int
		window size in seconds that will determine how long the preceding and following features will be
	idx : int (optional)
		index of the file to process, only used for verbose and can be usefull when multiprocessing is one
	total : int (optional)
		total number of fies to be processed, only used for verbose and can be usefull when multiprocessing is one
	save_location : os.path
		folder location where to save the labels to for each subject together with type, window, subject, label, index, and counter information


	Returns
	---------
	data : dict()
		dictionary that will hold episodes by label
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
		start_index = row.loc['start_index']
		stop_index = row.loc['stop_index']
		
		# note that we flip the encoding here. so labels with 0 become 1, and labels that are 1 become zero. This is basically changing how we define the positive class
		label = 1 - row.loc['label']
		
		# forward search to extend stop index
		stop_index = _forward_search_episode(actigraph_acc, stop_index, hz = 100, max_search_min = 5, std_threshold = 0.004, verbose = False)
		# backwar search to extend start index
		start_index = _backward_search_episode(actigraph_acc, start_index, hz = 100, max_search_min = 5, std_threshold = 0.004, verbose = False)

		
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
def batch_process_get_nw_time_from_raw(hdf5_acc_file, hdf5_nw_file, limit = None, skip_n = 0, hz = 100):
	"""
	Grid search approach to calculate non-wear time from raw data by using the CNN model and trying out different hyperparameter values

	Parameters
	-----------
	hdf5_read_file : os.path
		location of HDF5 file that contains the raw activity data for actigraph and actiwave
	hdf5_nw_file : os.path
		location of HDF5 file where to save the inferred non-wear to
	limit : int (optional)
		limit the number of subjects to be processed
	skip_n : int (optional)
		skip first N subjects
	hz : int (optional)
		sampling frequency of the raw acceleration data (defaults to 100HZ)
	"""

	# which CNN model to use
	model_folder = os.path.join('files', 'models', 'nw', 'cnn1d_60_20_20_early_stopping')
	# define architecture type
	cnn_type = 'v2'
	# define window length
	episode_window_sec = 3

	"""
		GRID SEARCH PARAMETERS
	"""
	# standard deviation threshold
	std_range = [0.004]
	# default classification when an episode does not have a starting or stop feature window (happens at t=0 or at the end of the data)
	edge_true_or_false_range = [True, False]
	# logical operator to see if both sides need to be classified as non-wear time (AND) or just a single side (OR)
	start_stop_label_decision_range = ['or', 'and']
	# merging of two candidate non-wear episodes that are 'distance_in_min_range' minutes apart from each other
	distance_in_min_range = [1, 2, 3, 4, 5]

	# loop over each standard deviation
	for std_threshold in std_range:
		# loop over each default setting
		for edge_true_or_false in edge_true_or_false_range:
			# loop over logical operator
			for start_stop_label_decision in start_stop_label_decision_range:
				# loop over each merging distance
				for distance_in_min in distance_in_min_range:

					# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
					subjects = get_all_subjects_hdf5(hdf5_file = hdf5_acc_file)[0 + skip_n:limit]

					# exclude subjects that have issues with their data
					subjects = [s for s in subjects if s not in get_subjects_with_invalid_data()]

					# use parallel processing to speed up processing time
					executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
					# create tasks so we can execute them in parallel
					tasks = (delayed(process_get_nw_time_from_raw)(subject, model_folder, hdf5_acc_file, hdf5_nw_file, cnn_type, std_threshold, episode_window_sec, start_stop_label_decision, distance_in_min, edge_true_or_false, hz, i, len(subjects)) for i, subject in enumerate(subjects))
					# execute task
					executor(tasks)

def	process_get_nw_time_from_raw(subject, model_folder, hdf5_read_file, hdf5_save_file, cnn_type, std_threshold, episode_window_sec, start_stop_label_decision, distance_in_min, edge_true_or_false, hz, idx, total):
	"""
	Get non-wear time from raw acceleration data by using a trained cnn model

	
	Parameters
	-----------
	subject : string
		subject ID to process
	model_folder : os.path
		folder location where different CNN models are stored
	hdf5_read_file : os.path
		file location of HDF5 that contains raw acceleration data
	hdf5_save_file : os.path
		file location where to save the inferred non-wear time to
	cnn_type : string
		what type of CNN architecture to use (v1, v2, v3 or v4 for example)
	std_threshold : float
		standard deviation threshold
	episode_window_sec : int
		define window length (for example 3 seconds)
	start_stop_label_decision : string
		logical operator to see if both sides need to be classified as non-wear time (AND) or just a single side (OR)
	distance_in_min : int
		number of minutes that will be used to merg two candidate non-wear episodes that are 'distance_in_min_range' minutes apart from each other
	edge_true_or_false : Boolen
		default classification when an episode does not have a starting or stop feature window (happens at t=0 or at the end of the data)
	hz : int
		sample frequency of the raw data
	idx : int (optional)
		index of the file to process, only used for verbose and can be usefull when multiprocessing is one
	total : int (optional)
		total number of files to be processed, only used for verbose and can be usefull when multiprocessing is one

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
	"""
	Calculate the classification performance metrics of several CNN models

	Parameters
	----------
	hdf5_read_file : os.path
		file location of the HDF5 file that contains the inferred non-wear time data and true non-wear time
		Note that the function batch_process_get_nw_time_from_raw will need to be executed first to obtain inferred non-wear time vectors
		This function only calculates performance measures based on the inferred and true non-wear time
	"""

	# define architecture type
	cnn_type = 'v2'
	# define window length
	episode_window_sec = 3

	"""
		PARAMETERS
	"""
	# standard deviation threshold
	std_threshold_range = [0.004]
	# default classification when an episode does not have a starting or stop feature window (happens at t=0 or at the end of the data)
	edge_true_or_false_range = [True, False]
	# logical operator to see if both sides need to be classified as non-wear time (AND) or just a single side (OR)
	start_stop_label_decision_range = ['or', 'and']
	# merging of two candidate non-wear episodes that are 'distance_in_min_range' minutes apart from each other
	distance_in_min_range = [1, 2, 3, 4, 5]

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

					# create subfolder based on cnn type and seconds used
					subfolder = f'{cnn_type}_{episode_window_sec}'

					# create subfolder if not exists
					create_directory(os.path.join('files', 'cnn_nw_performance', subfolder))

					# save to CSV
					all_results.to_csv(os.path.join('files', 'cnn_nw_performance', subfolder, f'{inferred_nw_time_group_name}_per_subject.csv'))
					
					
					tn = all_results['tn'].sum()
					fp = all_results['fp'].sum()
					fn = all_results['fn'].sum()
					tp = all_results['tp'].sum()

					logging.debug('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

					# calculate classification performance such as precision, recall, f1 etc.
					classification_performance = calculate_classification_performance(tn, fp, fn, tp)


					df_classification_performance = pd.DataFrame(pd.Series(classification_performance))
					df_classification_performance.to_csv(os.path.join('files', 'cnn_nw_performance', subfolder, f'{inferred_nw_time_group_name}_all_subjects.csv'))


"""
	EVALUATE BASELINE MODELS
"""

def batch_evaluate_baseline_models(hdf5_acc_file, hdf5_nw_file):
	"""
	Calculate non-wear time with two baseline algorithms. See paper description below

	Description from paper:
	Our proposed non-wear algorithm was compared to several baseline algorithms and existing non-wear detection algorithms to evaluate its 
	performance (van Hees et al., 2011, 2013) These baseline algorithms employ a similar analytical approach commonly found in count-based 
	algorithms(L. Choi et al., 2011; Hecht et al., 2009; Troiano et al., 2007), that is, detecting episodes of no activity by using an interval 
	of varying length. The first baseline algorithm detected episodes of no activity when the raw acceleration data was below a SD threshold 
	of 0.004g, 0.005g, 0.006g, and 0.007g and the duration did not exceed an interval length of 15, 30, 45, 60, 75, 90, 105, or 120 minutes. 
	A similar approach was proposed in another recent study as the SD_XYZ method,(Ahmadi et al., 2020) although the authors fixed the 
	threshold to 13mg and the interval to 30 minutes for a wrist worn accelerometer. Throughout this paper, the first baseline algorithm is 
	referred to as the XYZ algorithm. The second baseline algorithm was similar to the first baseline algorithm, albeit that the SD threshold 
	was applied to the vector magnitude unit (VMU) of the three axes, where VMU is calculated as √(〖acc〗_x^2+〖acc〗_y^2+ 〖acc〗_z^2  ), 
	with accx, accy, and accz referring to each of the orthogonal axes. A similar approach has recently been proposed as the SD_VMU 
	algorithm (Ahmadi et al., 2020). Throughout this paper, this baseline algorithm is referred to as the VMU algorithm. 
	
	Parameters
	-------------
	hdf5_acc_file
		file location of the HDF5 file that contains the raw acceleration data
	hdf5_nw_file : os.path
		file location of the HDF5 file that contains the inferred non-wear time data and true non-wear time
	"""

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
	"""
	Function that is part of the batch_evaluate_baseline_models function.
	Here we calculate the performance of a single parameterized baseline model.

	Parameters
	------------
	subjects : list
		list of all subject ID that we want to include when calculating the performance of this baseline algorithm. Typically all available subjects will be used
	hdf5_acc_file
		file location of the HDF5 file that contains the raw acceleration data
	hdf5_nw_file : os.path
		file location of the HDF5 file that contains the inferred non-wear time data and true non-wear time
	std_threshold : float
		standard deviation threshold used inside this baseline algorithm
	episode_length : int
		this can be seen as the interval. The interval is used as a minimum window in which the 'std_threshold' need to be below in order to classify as non-wear time.
	"""

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
	
	# count occurences of true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp)
	tn = all_results['tn'].sum()
	fp = all_results['fp'].sum()
	fn = all_results['fn'].sum()
	tp = all_results['tp'].sum()

	logging.debug('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

	# calculate classification performance such as precision, recall, f1 etc.
	classification_performance = calculate_classification_performance(tn, fp, fn, tp)

	# create dataframe
	df_classification_performance = pd.DataFrame(pd.Series(classification_performance))
	# store dataframe as CSV file
	df_classification_performance.to_csv(os.path.join(save_folder, f'{std_threshold}_{episode_length}_all.csv'))

"""
	FUNCTIONS THAT WILL CREATE PLOTS THAT ARE USED WITHIN THE PAPER:
	A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks
"""

def process_plot_start_stop_segments(merged_episodes_folder, hdf5_acc_file, plot_folder, hz = 100, std_threshold = 0.004):
	"""
	Start or the stop segments of candidate non-wear episodes where features of a length of 2-10 seconds were extracted
	This basically shows raw acceleration data of candidate non-wear episodes from where we extracted preceding and following features

	[FIGURE 1] Start or the stop segments of candidate non-wear episodes where features of a length of 2-10 seconds were extracted; 
	(a) start or stop episodes of true non-wear time, (b) start or stop episodes of wear time.
	
	Parameters
	-----------
	merged_episodes_folder : os.path
		folder location that contain candidate non-wear episodes with start and stop timestamps (these have been merged, meaning, that two episodes
		in close proximity have been merged together to from a larger one)
	hdf5_acc_file : os.path
		file location of the HDF5 file that contains the raw acceleration data
	hz : int (optional)
		sample frequency of the data. Basically to know how many data samples we have within a single second. Defaults to 100Hz
	std_threshold : float (optional)
		standard deviation threshold to find candidate non-wear episodes. This is used to extend the edges of an episode to go from 1-min resolution to 1-sec resolution
	"""

	# dictionary to hold plot data
	plot_data = {'0' : [], '1' : []}

	# define the length of the window in seconds
	episode_window_sec = 20 * hz
	# define how much of the flat line nees to be shown in the plot
	show_flat_sec = 20 * hz

	# read csv files in merged episodes folder and loop over each file in F
	for f in read_directory(merged_episodes_folder):

		# we limit since we don't want to plot more than 20 of each class type
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

			# get the start timestamp
			start_index = row.loc['start_index']
			# get the stop timestamp
			stop_index = row.loc['stop_index']
			# get the label to know if it's non-wear time, or if its wear time
			label = row.loc['label']
			
			# forward search to extend stop index (to get a 1-sec resolution)
			stop_index = _forward_search_episode(actigraph_acc, stop_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = False)
			# backwar search to extend start index
			start_index = _backward_search_episode(actigraph_acc, start_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = False)

			# get start episode
			start_episode = actigraph_acc[start_index - episode_window_sec : start_index + show_flat_sec]
			# get stop episode
			stop_episode = actigraph_acc[stop_index - show_flat_sec: stop_index + episode_window_sec]
			
			# here we choose to show only the episodes that show large standard deviation within the acceleration data.
			# this is just so we have nicer plots, rather then plotting the ones which show very flat lines
			if start_episode.shape[0] == episode_window_sec + show_flat_sec:
				if np.all(np.std(start_episode, axis = 0) > 0.1):
					plot_data[f'{label}'].append((subject, start_episode))				
			if stop_episode.shape[0] == episode_window_sec + show_flat_sec:
				if np.all(np.std(stop_episode, axis = 0) > 0.1):
					plot_data[f'{label}'].append((subject, stop_episode))

	# call plot function to plot the data
	plot_start_stop_segments(plot_data, plot_folder)

def process_plot_cnn_classification_performance(model_folder, plot_folder):
	"""
	Create four heatmaps that show the classification performance metrics for the training, validation, and test set of the CNN model

	[Figure 3] Accuracy, precision, recall, and F1 performance metrics for training data (60%), validation data (20%), 
	and test data (20%) for the four architectures evaluated. All CNN models were trained for a total of 250 epochs with 
	early stopping enabled, a patience of 250 epochs, and restoring of the best weights when the validation loss was the 
	lowest. See the TensorFlow documentation for more info at https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping.
	
	Parameters
	------------
	model_folder : os.path
		folder location that contains the trained CNN models
	plot_folder : os.path
		folder location where the heatmap plot needs to be saved to
	"""

	# define the different CNN architectures that need to be plotted
	cnn_types = ['v1', 'v2', 'v3', 'v4']
	plot_data = {x : None for x in cnn_types}
	
	# which windows to show
	episode_window_sec = [2, 3, 4, 5, 6, 7, 8, 9, 10]

	# dictionary that holds the number of epochs the model trained for
	num_epochs = {f'{cnn}_{sec}' : 0 for cnn in cnn_types for sec in episode_window_sec}

	# the key of the dictionary is the column we want to include in the plot, the value is a translation we use (this translation will be used within the plot, rather
	# then the column name that contains underscores etc.)
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

	# loop over each cnn architecture type
	for cnn_type in cnn_types:

		# empty dataframe to hold the plot data for the cnn_type
		data = pd.DataFrame()

		# loop over each window length (this defines the preceding and following segment length)
		for ew in episode_window_sec:

			# dynamically create training history file name based on cnn_type and episode window
			training_history = os.path.join(model_folder, cnn_type, f'cnn_{cnn_type}_{ew}.h5_history.csv')
			test_history = os.path.join(model_folder, cnn_type, f'cnn_{cnn_type}_{ew}.h5_history_test.csv')

			# load data as dataframe
			df_training = pd.read_csv(training_history)
			df_test = pd.read_csv(test_history, index_col = 0)

			# number of epochs the model trained for
			epoch = df_training.iloc[-1].name + 1
			# add epoch to dictionary
			num_epochs[f'{cnn_type}_{ew}'] = epoch

			# add epoch row to all data dataframe
			data[ew] = pd.concat([df_training.iloc[-1], df_test.T.iloc[0]], axis = 0)
		
		# tranpose dataframe
		data = data.T

		# calculate training, dev, and test F1 score
		data['F1 train'] = 2 * data['tp'] / (2 * data['tp'] + data['fp'] + data['fn'])
		data['F1 val'] = 2 * data['val_tp'] / (2 * data['val_tp'] + data['val_fp'] + data['val_fn'])
		data['F1 test'] = 2 * data['test_tp'] / (2 * data['test_tp'] + data['test_fp'] + data['test_fn'])

		# take only the keys within the columns dictionary (so we could filter and only show the columns we want to)
		data = data[[x for x in columns.keys()]]
		# change the column name into a more human readable
		data = data.rename(columns = columns)

		# add a transposed version of the data dataframe to plot_data that holds the plot data for each cnn architecture
		plot_data[cnn_type] = data.T

	# call plot function to create the heatmap
	plot_cnn_classification_performance(plot_data = plot_data, plot_folder = plot_folder)

def process_plot_baseline_performance(data_folder, plot_folder):
	"""
	Create heatmaps that show the classification performance of the baseline algorithms. We create three plots here:
		- The F1 classification performance of the XYZ baseline algorithm (left), and the VMU baseline algorithm (right). 
		- The accuracy, precision, recall, and F1 score for XYZ baseline
		- The accuracy, precision, recall, and F1 score for VMU baseline
	
	[Figure 4] The F1 classification performance of the XYZ baseline algorithm (left), and the VMU baseline algorithm (right). 
	Note that a SD threshold of 0.003g performs poorly as it is below the accelerometer noise level and is therefore not 
	shown. See Figure S4 and S5 in the Supplementary Information for accuracy, precision, and recall scores.

	Parameters
	------------
	data_folder : os.path
		folder location that contain the classification performance of the baseline algorithms. Note that before we have those files, 
		the function 'batch_evaluate_baseline_models' needs to be executed first
	plot_folder : os.path
		folder location where to store the plots to
	"""

	# subfolders for both baseline algorithms
	subfolders = ['XYZ', 'VMU']

	# keep track of baseline F1 scores so we can also plot those side by side
	baseline_f1 = {x : None for x in subfolders}

	# only use the following metrics
	performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
	# standard deviation range
	std_threshold_range = [0.004, 0.005, 0.006, 0.007]
	# episode length
	episode_length_range = [15, 30, 45, 60, 75, 90, 105, 120]
	
	# create plot for each baseline
	for subfolder in subfolders:

		# create dictionary with metrics and dataframe to populate data to
		plot_data = {x : pd.DataFrame(columns = episode_length_range, index = std_threshold_range) for x in performance_metrics}

		# get all files that contain classification performance of all subjects (note the we filter for the file name that contains 'all' since 
		# this file contains the combined performance metrics of all subjects)
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
		
		# call plot function to create the heatmap for baseline algorithm
		plot_baseline_performance(plot_data, plot_name = subfolder, plot_folder = plot_folder)

		# save F1 scores
		baseline_f1[subfolder] = plot_data['f1']
	
	# call plot function to create a heatmap of only the F1 scores and combine both baseline algorithms on the same
	plot_baseline_performance_compare_f1(plot_data = baseline_f1, plot_name = 'baseline_comparison_f1.pdf', plot_folder = plot_folder)

def process_plot_overview_all_raw_nw_methods(cnn_csv_folder, cnn_type, episode_window, baseline_files, hees_files, plot_folder):
	"""
	Plot barchart to show the classification results of baselines methods, hees method, and the CNN method

	[FIGURE 5] A comparison of the classification performance metrics of the best performing baseline models XYZ_90 
	(i.e. calculating the standard deviation of the three individual axes and an interval length of 90 minutes), 
	VMU_105 (i.e. calculating the standard deviation of the VMU and an interval length of 105 minutes), the HEES_30 
	algorithm with a 30 minutes interval, the HEES_60 with a 60 minutes interval, the HEES_135 with tuned hyperparameters 
	and a 135 minutes interval, and the proposed CNN algorithm. Error bars represent the 95% confidence interval.

	Parameters
	-----------
	cnn_csv_folder : os.path
		folder location that contains the classification results of the cnn model (here we extract the best scores from)
	cnn_type : string
		the cnn architecture to use here, can be v1, v2, v3 or v4 (the best one is v2)
	episode_window : int
		length of the window from where we extracted preceding and following segments. Best is 3 seconds with the v2 CNN model
	baseline_files : os.path
		folder location where the baseline classification results are stored
	hees_files : os.path
		folder location where the hees classification results are stored
	plot_folder : os.path
		folder location where to save the plot to
	"""

	# dictionary to store all data
	data = {}

	"""
		OBTAIN PERFORMANCE OF BEST CNN MODEL
	"""
	cnn_classification_file = os.path.join(cnn_csv_folder, f'performance_cnn_{cnn_type}_{episode_window}_parameters.csv')
	# read csv file as pandas dataframe
	cnn_data = pd.read_csv(cnn_classification_file, index_col = 0)
	# sort by f1 score and take top score
	cnn_data = cnn_data.sort_values(by=['f1'], ascending = False).iloc[0]
	# add to dictionary
	data['cnn'] = {	'distance_in_min' : cnn_data.loc['distance_in_min'],
					'start_stop_label_decision' : cnn_data.loc['start_stop_label_decision'],
					'edge_true_or_false' : cnn_data.loc['edge_true_or_false'],
					'accuracy' : cnn_data.loc['accuracy'],
					'precision' : cnn_data.loc['precision'],
					'recall' : cnn_data.loc['recall'],
					'f1' : cnn_data.loc['f1']
					}
	logging.info(f"CNN top F1 : {data['cnn']['f1']}")

	"""
		OBTAIN HEES SCORES
	"""
	for f in read_directory(hees_files):
		
		# load file via pickle
		f_data = load_pickle(file_name = f)
		
		for key, value in f_data.items():

			# unpack key into variables (e.g. 135-1-8-2-1-1)
			mw, wo, st, sa, vt, va = key.split('-')
			
			data[f'hees_{mw}'] = {	'f1' : value['f1'],
									'precision' :  value['precision'],
									'recall' : value['recall'],
									'mw' : mw, 
									'wo' : wo, 
									'st' : st, 
									'sa' : sa, 
									'vt' : vt, 
									'va' : va}

			logging.info(f'Hees minimum window {mw} F1 : {value["f1"]}')
	"""
		OBTAIN BASELINE SCORES
	"""
	# obtain highest F1 score for baseline model without VMU
	for subfolder in ['VMU', 'NO_VMU']:	

		# keep track of highest f1 score
		top_f1 = 0
		top_f1_precision, top_f1_recall = 0, 0
		top_std = 0
		top_episode = 0

		for f in [x for x in read_directory(os.path.join(baseline_files, subfolder)) if 'all' in x]:

			# load file as dataframe, and take first columns, which makes it a seriers
			f_data = pd.read_csv(f, index_col = 0)['0']

			# get the F1 score
			f1_score = f_data.loc['f1']
			
			# update top score only if F1 is higher
			if f1_score > top_f1:
				top_f1 = f1_score
				top_f1_precision, top_f1_recall = f_data.loc['precision'], f_data.loc['recall']
				top_std = float(f.split(os.sep)[-1].split('_')[0])
				top_episode = int(f.split(os.sep)[-1].split('_')[1]) 
		
		logging.info(f'Top baseline {subfolder}: F1 : {top_f1}, std : {top_std}, episode : {top_episode}')
		
		# add to dictionary
		data[subfolder.lower()] = {	'f1' : top_f1,
											'precision' : top_f1_precision,
											'recall' : top_f1_recall, 
											'std' : top_std, 
											'episode' : top_episode}

	# create bar chart with F1 classification scores side by side
	plot_overview_all_raw_nw_methods(plot_data = data, plot_name = 'overview_all_raw_nw_methods.pdf', plot_folder = plot_folder)

def process_create_table_performance_cnn_nw_method(data_folder, cnn_type, episode_window, save_folder):
	"""
	Create a table to show the classification performance of the CNN method when using different hyperparameter settings.

	[Table 1] The classification of accuracy, precision, recall, and F1 performance metrics when applying the new algorithm on 
	50% of the available data (n = 291/583) while exploring 20 combinations of hyperparameter values; 95% confidence intervals 
	are shown between parentheses. Merge (mins) = the merging of neighbouring candidate non-wear episodes to handle artificial 
	movement. Logical operator = ‘AND’ if both start and stop segments or ‘OR’ if only one side of a candidate non-wear episode 
	needs to be classified as true non-wear time to subsequently classify the candidate non-wear episode as an episode of 
	true non-wear time. Edge default = the default classification of a candidate non-wear episode that has no start or end 
	segment, such cases that occur right at the beginning or end of the acceleration data and default to wear or non-wear time.

	Parameters
	-----------
	data_folder : os.path
		folder location that contains the classification performance of different hyperparameter values. Note that these files are
		created when calling the function 'calculate_cnn_classification_performance'
	cnn_type : string
		the cnn architecture to use here, can be v1, v2, v3 or v4 (the best one is v2)
	episode_window : int
		length of the window from where we extracted preceding and following segments. Best is 3 seconds with the v2 CNN model
	save_folder : os.path
		folder location where to store the table to
	"""

	# read files in folder
	F = [x for x in read_directory(os.path.join(data_folder, f'{cnn_type}_{episode_window}')) if 'all' in x]

	# empty dataframe
	df = pd.DataFrame()
	
	# loop over each file
	for f in F:

		# extract variables from file name
		cnn_type, episode_window_sec, start_stop_label_decision, distance_in_min, std_threshold, edge_true_or_false, *_ = f.split(os.sep)[-1].split('_')
		
		# read data as series
		f_data = pd.read_csv(f, index_col = 0)['0']
		# add relevant data to series
		f_data['cnn_type'] = cnn_type
		f_data['episode_window_sec'] = episode_window_sec
		f_data['start_stop_label_decision'] = start_stop_label_decision
		f_data['distance_in_min'] = distance_in_min
		f_data['std_threshold'] = std_threshold
		f_data['edge_true_or_false'] = 'nw time' if eval(edge_true_or_false) else 'wear time'

		# add seriers to dataframe
		df[f'{start_stop_label_decision.upper()}, {distance_in_min} min, {edge_true_or_false}'] = f_data
	
	# transpose dataframe and sort columns by index
	df = df.T.sort_index()
	# only select certain columns
	df_filtered = df[['distance_in_min', 'start_stop_label_decision', 'edge_true_or_false', 'accuracy', 'precision', 'recall', 'f1']]
	# sort on f1 score
	df_filtered = df_filtered.sort_values(by=['f1'], ascending = False)
	# save as CSV
	df_filtered.to_csv(os.path.join(save_folder, f'performance_cnn_{cnn_type}_{episode_window}_parameters.csv'))
	
def process_plot_cnn_training_results_per_epoch(model_folder, episode_window, cnn_type, plot_folder):
	"""
	Create a line chart with the training and validation details per epoch. The details are stored within a history file when we created the CNN model.
	Typically, the metrics callback when creating the cnn model defines what training and validation details we have.  These are defined in the functions.dl_functions.py file.
	A copy is presented here

		METRICS = [
		keras.metrics.TruePositives(name='tp'),
		keras.metrics.FalsePositives(name='fp'),
		keras.metrics.TrueNegatives(name='tn'),
		keras.metrics.FalseNegatives(name='fn'), 
		keras.metrics.BinaryAccuracy(name='accuracy'),
		keras.metrics.Precision(name='precision'),
		keras.metrics.Recall(name='recall'),
		keras.metrics.AUC(name='auc'),

	See also Figure S3 of the supplementary material for paper::
		A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks

	Paramaters
	-----------
	model_folder : os.path
		folder location where the CNN models are stored (within this folder, we have also saved the history file that contains the metrics per epoch)
		if you have saved this somewhere else, then use this folder here.
	episode_window: int
		the length of the window/episode (there are different CNN models that used different window lenghts, here we can choose from which CNN model we want to plot the history from)
	cnn_type : string
		the cnn architecture for which we want to plot the history data per epoch
	plot_folder : os.path
		folder location where the plot should be saved to
	"""

	# history file (contains the data per epoch)
	# note that we dynamically create the file location for the specific cnn architecture and episode window
	history_file = os.path.join(model_folder, cnn_type, f'cnn_{cnn_type}_{episode_window}.h5_history.csv')

	# load CSV file as pandas dataframe
	data = pd.read_csv(history_file, index_col = 0)

	# add F1 results (since we did not calculate them when we created the CNN model)
	data['F1'] = 2 * data['tp'] / (2 * data['tp'] + data['fp'] + data['fn'])
	data['val_F1'] = 2 * data['val_tp'] / (2 * data['val_tp'] + data['val_fp'] + data['val_fn'])

	# call plot function to plot the line chart
	plot_training_results_per_epoch(data = data, plot_name = f'epoch_results_cnn_{cnn_type}_{episode_window}', plot_folder = plot_folder)

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

def _calculate_confusion_matrix(key, values, hz = 100, verbose = False):

	# unpack y and y_hat
	y = values['y']
	y_hat = values['y_hat']

	# downscale to second level
	y = y[::hz]
	y_hat = y_hat[::hz]

	# get confusion matrix values
	tn, fp, fn, tp = get_confusion_matrix(y, y_hat, labels = [0,1]).ravel()

	if verbose:
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
	env = 'local'
	
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
		# location where to save the plots to
		plot_folder = os.path.join('plots', 'Paper2')
	
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
	merge_close_episodes_from_training_data(read_folder = episodes_folder, save_folder = merged_episodes_folder)

	# 1b) calculate true non-wear time from labeled episodes
	process_calculate_true_nw_time_from_labeled_episodes(merged_episodes_folder = merged_episodes_folder, hdf5_read_file = actiwave_actigraph_mapping_hdf5_file, hdf5_save_file = nw_time_hdf5_file )

	# 1c) plot episodes, both normal and grouped
	process_plot_merged_episodes(episodes_folder = episodes_folder, grouped_episodes_folder = merged_episodes_folder, hdf5_file = actiwave_actigraph_mapping_hdf5_file)

	"""
		2) PERFORM GRID SEARCH FOR DIFFERENT SIZED FEATURE WINDOWS
	"""

	# 2) create cnn model with grid search
	perform_grid_search_1d_cnn(	label_folder = merged_episodes_folder, \
								hdf5_read_file = actiwave_actigraph_mapping_hdf5_file, \
								save_data_location = os.path.join(os.sep, 'users', 'shaheensyed', 'hdf5', 'start_stop_data'),\
								save_model_folder = os.path.join('files', 'models', 'nw', 'cnn1d_60_20_20_early_stopping'),\
								file_limit = None)

	"""
		3) CREATE NON-WEAR TIME BY USING CNN MODEL
	"""
	batch_process_get_nw_time_from_raw(hdf5_acc_file = actiwave_actigraph_mapping_hdf5_file, hdf5_nw_file = nw_time_hdf5_file)

	"""
		4) CALCULATE CLASSIFICATION PERFORMANCE CNN MODEL COMPARED TO TRUE NON-WEAR TIME
	"""
	calculate_cnn_classification_performance(hdf5_read_file = nw_time_hdf5_file)

	"""
		5) EVALUATE BASELINE MODELS
	"""
	batch_evaluate_baseline_models(hdf5_acc_file = actiwave_actigraph_mapping_hdf5_file, hdf5_nw_file = nw_time_hdf5_file)
	
	"""
		6) PAPER PLOTS
	"""
	# Fig. 1 : Start or the stop segments of candidate non-wear episodes where features of a length of 2-10 seconds were extracted
	process_plot_start_stop_segments(merged_episodes_folder = merged_episodes_folder, hdf5_acc_file = actiwave_actigraph_mapping_hdf5_file, plot_folder = plot_folder)

	# Fig. 3 : Accuracy, precision, recall, and F1 performance metrics for training data (60%), validation data (20%), and test data (20%) for the four architectures evaluated. 
	process_plot_cnn_classification_performance(model_folder = os.path.join('files', 'models', 'nw', 'cnn1d_60_20_20_early_stopping'), plot_folder = plot_folder)

	# Fig 4, Fig S4, and Fig S5: Heatmaps to show the classification performance of the baseline algorithms
	process_plot_baseline_performance(data_folder = os.path.join('files', 'baseline_performance'), plot_folder = plot_folder)

	# Fig 5:  plot overview of all methods (cnn method, baseline method with and without VMU, and hees default and optimized)
	process_plot_overview_all_raw_nw_methods(cnn_csv_folder = plot_folder, cnn_type = 'v2', episode_window = 3, baseline_files = os.path.join('files', 'baseline_performance'), hees_files = os.path.join('files', 'grid-search-hees_original'), plot_folder = plot_folder)

	# Table 1 : Create a table to show the classification performance of the cnn nw method with different hyperparameter values
	process_create_table_performance_cnn_nw_method(data_folder = cnn_nw_classification_folder, cnn_type = 'v2', episode_window = 3, save_folder = plot_folder)

	# Fig S3: Create a line chart that shows the training and validation loss per epoch, including other metrics calculated during training
	process_plot_cnn_training_results_per_epoch(model_folder = os.path.join('files', 'models', 'nw', 'cnn1d_60_20_20_early_stopping'), cnn_type = 'v2', episode_window = 3, plot_folder = plot_folder)

	set_end(tic,process)

	