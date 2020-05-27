# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import norm
from math import sqrt

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import 	set_start, set_end, dictionary_values_bytes_to_string, create_directory, read_directory, calculate_vector_magnitude, \
										save_pickle, get_random_number_between, load_pickle, get_subject_counters_for_correction, get_subjects_with_invalid_data,\
										read_csv, convert_short_code_to_long
from functions.hdf5_functions import get_all_subjects_hdf5, read_metadata_from_group, read_dataset_from_group, read_metadata_from_group_dataset, save_multi_data_to_group_hdf5, save_meta_data_to_group_dataset, save_data_to_group_hdf5
from functions.actiwave_functions import create_actiwave_time_vector
from functions.plot_functions import plot_classification_results, plot_time_distribution, plot_nw_scenarios, plot_nw_distribution, plot_non_wear_data
from functions.gt3x_functions import rescale_log_data, create_time_array
from functions.raw_non_wear_functions import find_candidate_non_wear_segments_from_raw, find_consecutive_index_ranges
from functions.autocalibrate_functions_2 import get_calibration_weights, return_default_weights, find_segments_moving_averages
from functions.statistical_functions import calculate_lower_bound_Keogh, signal_to_noise
from functions.ml_functions import execute_gridsearch_cv, predict_class, get_confusion_matrix, calculate_classification_performance
from functions.datasets_functions import get_actigraph_acc_data, get_actiwave_acc_data, get_actiwave_hr_data, get_actiwave_ecg_data
from functions.signal_processing_functions import apply_butterworth_filter, resample_acceleration
from functions.dl_functions import train_mlp_classifier

"""
	GLOBAL VARIABLES
"""

# local
ACTIGRAPH_HDF5_FILE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIGRAPH_TU7.hdf5')
ACTIWAVE_HDF5_FILE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIWAVE_TU7.hdf5')
# ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')
ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join(os.sep, 'users', 'shaheensyed', 'hdf5', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')
# server
# ACTIGRAPH_HDF5_FILE = os.path.join(os.sep, 'media', 'shaheen', 'LaCie_server', 'ACTIGRAPH_TU7.hdf5')
# ACTIWAVE_HDF5_FILE = os.path.join(os.sep, 'media', 'shaheen',  'LaCie_server', 'ACTIWAVE_TU7.hdf5')
# ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join(os.sep, 'media', 'shaheen', 'LaCie_server', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')

def batch_process_mapping_actiwave_on_actigraph(use_parallel = True, num_jobs = cpu_count(), limit = None, skip_n = 0, skip_processed_subjects = True):
	"""
	- BATCH PROCESS FUNCTION
	This function maps/aligns data from two devices (actigraph and actiwave) based on their union over time.
	In other words, we find the signals from the two devices that were recorded at the same time (here based on the timestamp)
	In context: actigraph was worn on the hip, and actiwave was worn on the chest. Here we are aiming to map the two signals by timestamps

	Parameters
	-----------
	use_parallel = Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs = int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	limit : int (optional)
		limit the number of subjects to be processed
	skip_n : int (optional)
		skip first N subjects
	skip_processed_subjects : Boolean (optional)
		skip subjects that are already part of the target hdf5 file
	"""

	# get all the subjects that have actiwave data
	subjects = get_all_subjects_hdf5(hdf5_file = ACTIWAVE_HDF5_FILE)[0 + skip_n:limit]


	# retrieve subjects that have already been processed
	if skip_processed_subjects:

		# get subjects
		processed_subjects = get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

		# exclude subjects that have already been processed
		subjects = [x for x in subjects if x not in processed_subjects]

	# if use_parallel is set to True, then use parallelization to process all files
	if use_parallel:

		# verbose
		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(process_mapping_actiwave_on_actigraph)(subject = subject, idx = i, total = len(subjects)) for i, subject in enumerate(subjects))
		# execute task
		executor(tasks)

	else:

		# verbose
		logging.info('Processing one-by-one (parallelization off)')

		# loop over the subjects
		for i, subject in enumerate(subjects):

			# call function to map actiwave data onto actigraph data
			process_mapping_actiwave_on_actigraph(subject = subject, idx = i, total = len(subjects))


def process_mapping_actiwave_on_actigraph(subject, idx = 1, total = 1):
	"""
	Mapping actiwave data onto actigraph data, based on union of timestamps
	Allows analyzes of two-device-streams of data

	Paramaters
	----------
	subject: string
		subject ID of participant
	idx : int (optional)
		index of processed participant (just for the counter)
	total : int (optional)
		total of subjects to be processed (just for the counter)
	"""

	# verbose
	logging.info('{style} Processing subject:{} {}/{} {style}'.format(subject, idx, total, style = '='*10))

	# read meta data from the original EDF file that is stored as a dictionary in HDF5
	meta_data = read_metadata_from_group(group_name = subject, hdf5_file = ACTIWAVE_HDF5_FILE)
	# convert meta_data values from bytes to string
	meta_data = dictionary_values_bytes_to_string(meta_data)
	# read start of the measurement
	start_datetime = datetime.strptime(meta_data['Start Datetime'], '%Y-%m-%d %H:%M:%S')

	"""
		ECG DATA
	"""

	# read ecg data
	actiwave_ecg = read_dataset_from_group(group_name = subject, dataset = 'ecg', hdf5_file = ACTIWAVE_HDF5_FILE)
	# read ecg meta data
	actiwave_ecg_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'ecg', hdf5_file = ACTIWAVE_HDF5_FILE)
	# extract length of data
	ecg_length = actiwave_ecg_meta_data['NSamples']
	# extract sample frequency (hz)
	ecg_hz = int(actiwave_ecg_meta_data['Sample Frequency'])
	# create time vector
	actiwave_ecg_time = create_actiwave_time_vector(start_datetime = start_datetime, length = ecg_length, hz = ecg_hz)

	"""
		HEART RATE
	"""

	#read hr data
	actiwave_hr = read_dataset_from_group(group_name = subject, dataset = 'estimated_hr', hdf5_file = ACTIWAVE_HDF5_FILE)
	# read ecg meta data
	actiwave_hr_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'estimated_hr', hdf5_file = ACTIWAVE_HDF5_FILE)
	# extract length of data
	hr_length = actiwave_hr_meta_data['NSamples']
	# create time vector
	actiwave_hr_time = create_actiwave_time_vector(start_datetime = start_datetime, length = hr_length, hz = 1)


	"""
		ACCELERATION ACTIWAVE
	"""

	# read acceleration data
	actiwave_acc = read_dataset_from_group(group_name = subject, dataset = 'acceleration', hdf5_file = ACTIWAVE_HDF5_FILE)
	# # read ecg meta data
	actiwave_acc_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'acceleration', hdf5_file = ACTIWAVE_HDF5_FILE)
	# # extract length of data
	acc_length = actiwave_acc_meta_data['NSamples']
	# extract the sample frequency
	acc_hz = actiwave_acc_meta_data['Sample Frequency']
	# create time vector
	actiwave_time = create_actiwave_time_vector(start_datetime = start_datetime, length = acc_length, hz = acc_hz)

	"""
		ACCELERATION ACTIGRAPH
	"""

	# read log data
	actigraph_acc = read_dataset_from_group(group_name = subject, dataset = 'log', hdf5_file = ACTIGRAPH_HDF5_FILE)
	# check if subject had actigraph log data (this is the raw data extracted from the .gt3x file)
	if actigraph_acc is None:
		logging.warning('Subject {} has no Actigraph log data. Skipping ...'.format(subject))
		return
	# read log meta data
	actigraph_acc_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'log', hdf5_file = ACTIGRAPH_HDF5_FILE)
	# scale log data
	actigraph_acc = rescale_log_data(actigraph_acc, acceleration_scale = actigraph_acc_meta_data['Acceleration_Scale'])	
	# get time data
	actigraph_time = read_dataset_from_group(group_name = subject, dataset = 'time', hdf5_file = ACTIGRAPH_HDF5_FILE)
	# convert time data to correct time series array with correct miliseconds values
	actigraph_time = create_time_array(actigraph_time)
	

	"""
		GET START AND STOP SLICES
	"""

	# get start time from the actigraph signal
	start_time = actigraph_time[0]
	# get the end time from the actiwave signal
	end_time = np.array(actiwave_time[-1], dtype = 'datetime64[s]')

	# check if there are overlapping times
	if end_time < start_time:
		logging.error('Actiwave end time {} does not overlap with actigraph start time {}'.format(end_time, start_time))
		return

	# check if actigraph start time contains in actiwave signal, if not, we need to use the start time from the actiwave signal and not from the actigraph
	if start_time not in actiwave_time:
		start_time = actiwave_time[0]

	# get start and stop index for actiwave_acc
	acc_start_index = np.where(actiwave_time == start_time)[0][0]
	acc_end_index = np.where(actiwave_time == end_time)[0][0]
	
	# get start and stop index for actigraph acceleration
	acc_actigraph_start_index = np.where(actigraph_time == start_time)[0][0]
	acc_actigraph_end_index = np.where(actigraph_time == end_time)[0][0]
	
	# get start and stop index for actiwave estimated heart rate
	hr_start_index = np.where(actiwave_hr_time == start_time)[0][0]
	hr_end_index = np.where(actiwave_hr_time == end_time)[0][0]

	# get start and stop index for ECG data
	ecg_start_index = np.where(actiwave_ecg_time == start_time)[0][0]
	ecg_end_index = np.where(actiwave_ecg_time == end_time)[0][0]

	"""
		SLICE THE DATA
	"""

	# slice heart rate data
	actiwave_hr, actiwave_hr_time = actiwave_hr[hr_start_index:hr_end_index], actiwave_hr_time[hr_start_index:hr_end_index]
	# slice actiwave acceleration data
	actiwave_acc, actiwave_time = actiwave_acc[acc_start_index:acc_end_index], actiwave_time[acc_start_index:acc_end_index]
	# slice actigraph acceleration
	actigraph_acc, actigraph_time = actigraph_acc[acc_actigraph_start_index:acc_actigraph_end_index], actigraph_time[acc_actigraph_start_index:acc_actigraph_end_index]
	# slice ECG data
	actiwave_ecg, actiwave_ecg_time = actiwave_ecg[ecg_start_index:ecg_end_index], actiwave_ecg_time[ecg_start_index:ecg_end_index]


	"""
		SAVE DATA TO HDF5
	"""

	# create list of data, data_name, and meta_data. IMPORTANT, mapping is based on position within the three lists, so they need to be of equal length with corresponding index number
	data = [actigraph_acc, actigraph_time.astype('int64'), actiwave_acc, actiwave_time.astype('int64'), actiwave_hr, actiwave_hr_time.astype('int64'), actiwave_ecg, actiwave_ecg_time.astype('int64')]
	data_name = ['actigraph_acc', 'actigraph_time', 'actiwave_acc', 'actiwave_time', 'actiwave_hr', 'actiwave_hr_time', 'actiwave_ecg', 'actiwave_ecg_time']
	meta_data = [actigraph_acc_meta_data, None, actiwave_acc_meta_data, None, actiwave_hr_meta_data, None, actiwave_ecg_meta_data, None]

	# check if lists are of equal length
	if any(len(x) != len(data) for x in [data_name, meta_data]):
		logging.error('Lists data, data_name, and meta_data are not all of equal length')
		exit(1)

	# save all lists of data to HDF5 file
	save_multi_data_to_group_hdf5(group = subject, data = data, data_name = data_name, meta_data = meta_data, overwrite = True, create_group_if_not_exists = True, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)


def batch_process_calculate_autocalibrate_weights(use_parallel = True, num_jobs = cpu_count(), limit = None, skip_n = 0, skip_processed_subjects = True):
	"""
	Batch processing to find autocalibration weights for Y, X, and Z acceleration including bias

	Parameters
	-----------
	use_parallel = Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs = int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	limit : int (optional)
		limit the number of subjects to be processed
	skipN : int (optional)
		skip first N subjects
	"""

	# get all the subjects that have actiwave data
	subjects = get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)[0 + skip_n:limit]

	# skip subjects that already have autocorrelation weights
	if skip_processed_subjects:

		# retrieve all the subjects that already have the t_y weight (t_y could also be any other of the weights variables)
		processed_subjects = [x for x in subjects if read_metadata_from_group_dataset(group_name = x, dataset = 'actigraph_acc', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE).get('t_y') is not None]
		# exclude from subjects
		subjects = [x for x in subjects if x not in processed_subjects]


	# if use_parallel is set to True, then use parallelization to process all files
	if use_parallel:

		# verbose
		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(process_calculate_autocalibrate_weights)(subject = subject, idx = i, total = len(subjects)) for i, subject in enumerate(subjects))
		# execute task
		executor(tasks)

	else:

		# verbose
		logging.info('Processing one-by-one (parallelization off)')

		# loop over the subjects
		for i, subject in enumerate(subjects):

			process_calculate_autocalibrate_weights(subject = subject, idx = i, total = len(subjects))


def process_calculate_autocalibrate_weights(subject, idx, total):
	"""
	Calculate autocalibrate weights
	- theta_y, theta_x, theta_z
	- bias_y, bias_x, bias_z

	For actigraph as well as actiwave data

	Paramaters
	----------
	subject: string
		subject ID of participant
	idx : int
		index of processed participant (just for the counter)
	total : int
		total of subjects to be processed (just for the counter)
	"""

	# verbose
	logging.info('Processing subject:{} {}/{}'.format(subject, idx, total))

	"""
		ACTIGRAPH AUTOCALIBRATION
	"""

	# read actigraph acceleration data
	actigraph_acc = read_dataset_from_group(group_name = subject, dataset = 'actigraph_acc', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	# read actigraph acceleration meta data
	actigraph_acc_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'actigraph_acc', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	
	# get the sample rate (hz = number of measurements per second)
	actigraph_sample_rate = int(actigraph_acc_meta_data['Sample_Rate'])
	
	# find segments with no movement based on standard deviation threshold
	actigraph_segments = find_segments_moving_averages(actigraph_acc, std_threshold = 0.004, hz = actigraph_sample_rate)

	# check if there are non-movement segments
	if len(actigraph_segments) > 0:

		"""
			See paper: vincent van hees
			To ensure a meaningful and robust autocalibration, it was only executed when the 
			calibration ellipsoid was sufficiently sparsely populated with data points (calibration epochs). 
			For this evaluation, we used a sparseness criteria of at least one ellipsoid value higher 
			than 300 mg and at least one value lower than 􏰅300 mg for each of the three sensor axes.
		"""
		if np.any(np.any(actigraph_segments > .300, axis = 0), axis = 0) and np.any(np.any(actigraph_segments < .300, axis = 0), axis = 0):
			
			# calculate the weights
			actigraph_weights = get_calibration_weights(X_train = actigraph_segments)

		else:
			logging.warning('Actigraph segments do not meet sparsity criteria. Setting weights to default ({})'.format(subject))
			# set weights to default, no change when autocalibrate
			actigraph_weights = return_default_weights()
	else:
		logging.warning('Actigraph segments count = 0 ({})'.format(subject))
		# set weights to default, no change when autocalibrate
		actigraph_weights = return_default_weights()

	"""
		ACTIWAVE AUTOCALIBRATION
	"""

	# read actiwave acceleration data
	actiwave_acc = read_dataset_from_group(group_name = subject, dataset = 'actiwave_acc', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	# read actiwave acceleration meta data
	actiwave_acc_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'actiwave_acc', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	# get the sample rate (hz = number of measurements per second)
	actiwave_sample_rate = int(actiwave_acc_meta_data['Sample Frequency'])
	

	# find segments with no movement based on standard deviation threshold
	actiwave_segments = find_segments_moving_averages(actiwave_acc, std_threshold = 0.007, hz = actiwave_sample_rate)

	# check if there are non-movement segments
	if len(actiwave_segments) > 0:

		"""
			See paper: vincent van hees
			To ensure a meaningful and robust autocalibration, it was only executed when the 
			calibration ellipsoid was sufficiently sparsely populated with data points (calibration epochs). 
			For this evaluation, we used a sparseness criteria of at least one ellipsoid value higher 
			than 300 mg and at least one value lower than 􏰅300 mg for each of the three sensor axes.
		"""
		if np.all(np.any(actiwave_segments > .300, axis = 0), axis = 0) and np.all(np.any(actiwave_segments < .300, axis = 0), axis = 0):
			
			# calculate the weights
			actiwave_weights = get_calibration_weights(X_train = actiwave_segments)

		else:
			logging.warning('Actiwave segments do not meet sparsity criteria. Setting weights to default ({})'.format(subject))
			# set weights to default, no change when autocalibrate
			actiwave_weights = return_default_weights()
	else:
		logging.warning('Actiwave segments count = 0 ({})'.format(subject))
		# set weights to default, no change when autocalibrate
		actiwave_weights = return_default_weights()

	"""
		SAVE WEIGHTS
	"""

	# save actigraph weights
	save_meta_data_to_group_dataset(group_name = subject, dataset = 'actigraph_acc', meta_data = actigraph_weights, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	# save actiwave weights
	save_meta_data_to_group_dataset(group_name = subject, dataset = 'actiwave_acc', meta_data = actiwave_weights, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)


def batch_process_create_features_from_labels(function, limit = None, skip_n = 0, use_parallel = False, num_jobs = cpu_count(), 
												label_location = os.path.join('labels', 'actigraph-actiwave-mapping', 'non-wear-time-by-start-stop'), 
												save_location = os.path.join('labels', 'actigraph-actiwave-mapping', 'non-wear-time-features')):
	"""
	Create a descriptive feature dataset with labels for candidate segments. Here features are a number of descriptive statistics that characterize a candidate segment. For instance
	we calculate the mean, mode, median, standard deviation, kurosis etc. that will later be used to train a machine learning classifier.

	We read in all .csv files that contain the labeled data with start and stop times. Each csv file can be read in as a pandas dataframe with columns representing:
		counter : index of candidate segment non wear time
		label : 0 for non wear time and 1 for wear time
		start : start time of segment 
		stop : end time of segment

	Parameters
	-----------
	function : python function
		function to call to batch process the creation of features
	limit : int (optional)
		limit the number of subjects to be processed
	skip_n : int (optional)
		skip first N subjects
	use_parallel = Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs = int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	label_location : os.path
		folder location where the csv files are stored that contain the labeled data, that is, start/stop time segments with 0 for non wear time and 1 for wear time
	save_location : os.path
		location where to save the feature data
	"""

	# verbose
	logging.info('Start batch processing create features from labels')

	# change save location depending on called function. for create_ml_features_from_labels append 'ml' to the folder, for create_dl_features_from_labels append 'dl'
	if function.__name__ == 'create_ml_features_from_labels':
		save_location = os.path.join(save_location, 'ml')
	elif function.__name__ == 'create_dl_features_from_labels':
		save_location = os.path.join(save_location, 'dl')
	else:
		logging.error('Unknown function name : {}'.format(function.__name__))
		exit(1)
	
	# create save_location if not exist already
	create_directory(save_location)

	# read the .csv files with the labeled data (note each .csv file can be read as a pandas dataframe)
	F = [f for f in read_directory(label_location) if f[-4:] == '.csv'][0 + skip_n:limit]

	# loop over the subjects
	if use_parallel:

		# verbose
		logging.info('Processing in parallel (parallelization on)')
		
		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(function)(f, save_location, i, len(F)) for i, f in enumerate(F))
		# execute task
		executor(tasks)

	else:

		# verbose
		logging.info('Processing one-by-one (parallelization off)')

		# loop over the subjects
		for i, f in enumerate(F):

			# create features from labeles
			function(f, save_location, i, len(F))


def create_ml_features_from_labels(f, save_location, i = 1, total = 1, autocalibrate = False, apply_filter = False, apply_resampling = True):
	"""
	Create a descriptive feature dataset with labels for candidate segments and save as pickle file (a single file per subject)

	Parameters
	----------
	f : string
		file location of .pkl file with start, stop, counter, and label information for each candidate segment
	save_location : os.path
		location where to save the feature data
	i : int (optional)
		index of number of processed file (only relevant when performing batch processing so we can keep track of the status)
	total : int (optional)
		total number of files to be processed (only relevant when performing batch processing so we can keep track of the status)
	autocalibrate: Boolean (optional)
		set to True if data needs to be autocalibrated first (make sure calibration weights are calculated first before doing this.)
	apply_filter : Boolean (optional)
		if set to True, then apply bandwith filter to acceleration data. Defaults to False
	apply_resampling : Boolean (optional)
		if set to True, then resample sampling frequency. For example, resample 100hz to 32hz to make frequencies comparable. Defaults to False
	"""

	logging.info('{style} Processing file: {} {}/{} {style}'.format(f, i, total, style = '='*10))
	
	# load the pandas dataframe from the csv file, and also transpose
	df = pd.read_csv(f, index_col = 0)

	# check if dataframe contains rows
	if len(df) == 0:
		logging.debug('Dataframe contains no rows, skipping....')
		return

	# extract subject ID from index, can be any index, but here we take the first index, note that the subject ID is the first 8 digits.
	subject = df.index[0][0:8]

	# new dataframe to populate later and store again
	df_new = pd.DataFrame()

	"""
		GET DATA
	"""

	# actigraph acceleration data
	actigraph_acc, _ , actigraph_time = get_actigraph_acc_data(subject, ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE, autocalibrate)
	# actiwave acceleration data
	actiwave_acc, _ , actiwave_time = get_actiwave_acc_data(subject, ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE, autocalibrate)
	# actiwave heart rate data
	actiwave_hr, actiwave_hr_time = get_actiwave_hr_data(subject, ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	
	"""
		RESAMPLE AND FILTER DATA
	"""

	# if apply_resampling is set to True, then resample original sample frequency to target frequency, so here 100hz to 32hz since actiwave uses 32 hz data
	if apply_resampling:

		# resample 1hz to 32hz
		actiwave_hr = actiwave_hr.repeat(32, axis = 0)

		# resample actigraph acceleration data from 100hz to 32 hz without parallel processing of the 3 axes
		actigraph_acc = resample_acceleration(actigraph_acc, 100, 32, False)

	if apply_filter:

		# apple butterworth bandwith filter to actigraph and actiwave acceleration data
		actigraph_acc = apply_butterworth_filter(actigraph_acc, n = 4, wn = np.array([0.01, 7]), btype = 'bandpass', hz = 32)
		actiwave_acc = apply_butterworth_filter(actiwave_acc, n = 4, wn = np.array([0.01, 7]), btype = 'bandpass', hz = 32)

	# check if both arrays have equal lengths
	if actigraph_acc.shape[0] != actiwave_acc.shape[0]:
		
		logging.warning('Actigraph and Actiwave signals have unequal number of samples. {} vs {}. Clipping on the lowest number'.format(actigraph_acc.shape[0], actiwave_acc.shape[0]))

		# get the clip length
		min_length = min(actigraph_acc.shape[0], actiwave_acc.shape[0])
		# clip actigraph acceleration
		actigraph_acc = actigraph_acc[: min_length]
		# clip actiwave acceleration 
		actiwave_acc = actiwave_acc[:min_length]
		# also for the actiwave heart rate
		actiwave_hr = actiwave_hr[:min_length]
		# and clip the time array
		actiwave_time = actiwave_time[:min_length]

	"""
		ADD VMU TO DATA
	"""

	# create actigraph VMU data
	actigraph_vmu = calculate_vector_magnitude(actigraph_acc, minus_one = True, round_negative_to_zero = True)
	# add VMU to actigraph acceleration
	actigraph_acc = np.hstack((actigraph_acc, actigraph_vmu))

	# create actigraph VMU data
	actiwave_vmu = calculate_vector_magnitude(actiwave_acc, minus_one = True, round_negative_to_zero = True)
	# add VMU to actigraph acceleration
	actiwave_acc = np.hstack((actiwave_acc, actiwave_vmu))


	"""
		CREATING THE DATAFRAMES
	"""

	if apply_resampling:
		# create dataframe of actigraph acceleration + non-wear vector
		df_actigraph_acc = pd.DataFrame(actigraph_acc, index = actiwave_time, columns = ['Y', 'X', 'Z', 'VMU ACTIGRAPH'])
	else:
		# create dataframe of actigraph acceleration + non-wear vector
		df_actigraph_acc = pd.DataFrame(actigraph_acc, index = actigraph_time, columns = ['Y', 'X', 'Z', 'VMU ACTIGRAPH'])

	# create dataframe of actiwave acceleration + VMU
	df_actiwave_acc = pd.DataFrame(actiwave_acc, index = actiwave_time, columns = ['Y', 'X', 'Z', 'VMU ACTIWAVE'])
	if apply_resampling:
		df_actiwave_hr = pd.DataFrame(actiwave_hr, index = actiwave_time, columns = ['HR ACTIWAVE'])
	else:
		# create dataframe of actiwave acceleration
		df_actiwave_hr = pd.DataFrame(actiwave_hr, index = actiwave_hr_time, columns = ['HR ACTIWAVE'])

	# loop trough the columns
	for index, row in df.iterrows():

		# get the start and stop values from the value componont
		start, stop, counter, label = row['start'], row['stop'], row['counter'], row['label']

		# convert start and stop to datetime64 object
		start = np.datetime64(start)
		stop = np.datetime64(stop)
		
		# obtain correct slice for actigraph data
		df_actigraph_segment = df_actigraph_acc.loc[start:stop]
		# obtain correct slice for actigraph data
		df_actiwave_segment = df_actiwave_acc.loc[start:stop]
		# obtain correct slice for actigraph heart rate data
		df_actiwave_hr_segment = df_actiwave_hr.loc[start:stop]

		# join the acceleration data
		joined_segment = df_actigraph_segment.iloc[:,[3]].join(df_actiwave_segment.iloc[:,[3]], how = 'inner')
		# add heart rate data
		joined_segment = joined_segment.join(df_actiwave_hr_segment.iloc[:,[0]], how = 'inner')

		# give easy variable name to column and drop NaNs
		a = joined_segment['VMU ACTIGRAPH'].dropna().values
		b = joined_segment['VMU ACTIWAVE'].dropna().values
		c = joined_segment['HR ACTIWAVE'].dropna().values

		# calculate descriptive features for candidate segment
		features = get_features_for_candidate_segment(a, b, c, None, start, stop, counter, label, for_training = True)		
		
		# add to new dataframe with correct column name
		df_new[index] = features

	# save new dataframe as csv
	df_new.to_csv(os.path.join(save_location, subject + '.csv'))

	# save new dataframe as pickle
	save_pickle(obj = df_new, file_name = subject, folder = save_location)


def create_dl_features_from_labels(f, save_location, i = 1, total = 1, autocalibrate = False, apply_resampling = True, resample_hz = 1, apply_filter = False):
	"""
	Create deep learning features from labels. Basically allowing the raw acceleration data to feed into a neural network

	Parameters
	----------
	f : string
		file location of .pkl file with start, stop, counter, and label information for each candidate segment
	save_location : os.path
		location where to save the feature data
	i : int (optional)
		index of number of processed file (only relevant when performing batch processing so we can keep track of the status)
	total : int (optional)
		total number of files to be processed (only relevant when performing batch processing so we can keep track of the status)
	autocalibrate: Boolean (optional)
		set to True if data needs to be autocalibrated first (make sure calibration weights are calculated first before doing this.)
	apply_resampling : Boolean (optional)
		if set to True, then resample sampling frequency. For example, resample 100hz to 32hz to make frequencies comparable. Defaults to False
	resample_hz : int (optional)
		if apply_resampling is set to True, then resample_hz determines to what frequency.  
	apply_filter : Boolean (optional)
		if set to True, then apply bandwith filter to acceleration data. Defaults to False
	"""

	logging.info('{style} Processing file: {} {}/{} {style}'.format(f, i, total, style = '='*10))
	
	# load the pandas dataframe from the csv file, and also transpose
	df = pd.read_csv(f, index_col = 0)

	# check if dataframe contains rows
	if len(df) == 0:
		logging.debug('Dataframe contains no rows, skipping....')
		return

	# extract subject ID from index, can be any index, but here we take the first index, note that the subject ID is the first 8 digits.
	subject = df.index[0][0:8]

	"""
		GET DATA
	"""
	# actigraph acceleration data
	actigraph_acc, *_ = get_actigraph_acc_data(subject, ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE, autocalibrate)
	# actiwave acceleration data
	actiwave_acc, _, actiwave_time = get_actiwave_acc_data(subject, ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE, autocalibrate)
	
	# if the resample_hz is lower than 32, meaning that we downscale the data, we need to downscale the 
	if resample_hz == 1:
		actiwave_time = actiwave_time[::32]
	elif resample_hz == 32:
		pass
	else:
		# extend time array from 32 samples per second to resample_hz samples per second.
		logging.error('Upscale/downscale of actiwave_time for resample_hz {} not yet implemented'.format(resample_hz))
		exit(1)

	# actiwave heart rate data
	actiwave_hr, _ = get_actiwave_hr_data(subject, ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	if resample_hz > 1:
		# actiwave_hr has 1 measurement per second, whereas we need to have 32 measurements per second, we simply extend the values
		actiwave_hr = actiwave_hr.repeat(resample_hz, axis = 0)

	# if apply_resampling is set to True, then resample original sample frequency to target frequency, so here 100hz to 32hz since actiwave uses 32 hz data
	if apply_resampling:

		actigraph_acc = resample_acceleration(actigraph_acc, 100, resample_hz, False)
		actiwave_acc = resample_acceleration(actiwave_acc, 32, resample_hz, False)

	# if apply_filter is set to True, then apply a 4th degree butterworth filter
	if apply_filter:

		actigraph_acc = apply_butterworth_filter(actigraph_acc, n = 4, wn = np.array([0.01, 7]), btype = 'bandpass', hz = resample_hz)
		actiwave_acc = apply_butterworth_filter(actiwave_acc, n = 4, wn = np.array([0.01, 7]), btype = 'bandpass', hz = resample_hz)

	# check if both arrays have equal lengths
	if actigraph_acc.shape[0] != actiwave_acc.shape[0]:
		
		logging.warning('Actigraph and Actiwave signals have unequal number of samples. {} vs {}. Clipping on the lowest number'.format(actigraph_acc.shape[0], actiwave_acc.shape[0]))

		# get the clip length
		min_length = min(actigraph_acc.shape[0], actiwave_acc.shape[0])
		# clip actigraph acceleration
		actigraph_acc = actigraph_acc[:min_length]
		# clip actiwave acceleration 
		actiwave_acc = actiwave_acc[:min_length]
		# also for the actiwave heart rate
		actiwave_hr = actiwave_hr[:min_length]
		# and clip the time array
		actiwave_time = actiwave_time[:min_length]


	# create dataframe of actigraph acceleration
	df_actigraph_acc = pd.DataFrame(actigraph_acc, index = actiwave_time, columns = ['Y - ACTIGRAPH', 'X - ACTIGRAPH', 'Z - ACTIGRAPH'])
	# create dataframe of actiwave acceleration
	df_actiwave_acc = pd.DataFrame(actiwave_acc, index = actiwave_time, columns = ['Y - ACTIWAVE', 'X - ACTIWAVE', 'Z - ACTIWAVE'])
	# create dataframe of actiwave heart rate data
	df_actiwave_hr = pd.DataFrame(actiwave_hr, index = actiwave_time, columns = ['HR - ACTIWAVE'])

	# join the three dataframes
	df_joined = df_actigraph_acc.join(df_actiwave_acc, how = 'inner').join(df_actiwave_hr, how = 'inner')

	# empty lists to hold numpy array
	X, Y = [], []

	# loop trough the columns
	for _, row in df.iterrows():

		# get the start and stop values from the value componont
		start, stop, _ , label = row['start'], row['stop'], row['counter'], row['label']

		# convert start and stop to datetime64 object
		start = np.datetime64(start)
		stop = np.datetime64(stop)

		# verbose
		logging.debug('Start : {}, Stop : {}, Label : {}'.format(start, stop, label))

		# obtain correct slice for actigraph data
		df_segment = df_joined.loc[start:stop]

		# convert to numpy array
		np_segment = np.array(df_segment)

		# how many samples in 1 feature vector ( here we take 32 since we have 32 mesurements for each second)
		num_samples = resample_hz
		
		# check how many times feature samples can be extracted, we take 1 feature sample for every 32 measurements
		size = np_segment.shape[0] // num_samples

		# empty array for x values
		x = np.empty((size, num_samples, np_segment.shape[1]), dtype = 'float32')

		# create X values
		for i in range(0, size):

			# start slice
			start_slice = i * num_samples
			end_slice = start_slice + num_samples

			# append to x
			x[i] = np_segment[start_slice:end_slice]

		# create y array (these are the target labels)
		y = np.full((size, 1), label)

		logging.debug('Created x array with shape: {}'.format(x.shape))
		logging.debug('Created y array with shape: {}'.format(y.shape))

		# append to X and Y lists
		X.append(x)
		Y.append(y)

	# vertical stack individual numpy arrays in list to a single numpy array
	X = np.vstack(X)
	Y = np.vstack(Y)

	# save X and Y
	np.save(os.path.join(save_location, '{}_X.npy'.format(subject)), X)
	np.save(os.path.join(save_location, '{}_Y.npy'.format(subject)), Y)


def create_ml_classifier(feature_location = os.path.join('labels', 'actigraph-actiwave-mapping', 'non-wear-time-features'), model_save_location = os.path.join('files', 'ml-models-nowac1', 'ml-32-nofilter-minus-round')):
	"""
	Create machine learning classifier that is able to classify candidate segments of non wear time into real non wear time or wear time
		- real non wear time there is a discrapancy between the actigraph signal and actiwave signal
		- wear time can be when there is a candidate segment but the actiwave has also very low activity, for instance during sleep or sedentary time, and then there is a low discrancy between the two signals

	Parameters
	----------
	feature_location: os.path
		location with the labeled features
	model_save_location: os.path
		folder where the ML models need to be saved
	"""

	# feature_folders = ['ml-32-filter-minus-round', 'ml-32-filter-nominus-noround', 'ml-32-nofilter-minus-round', 'ml-32-nofilter-nominus-noround']
	# feature_folders = ['ml-32-nofilter-minus-round']
	feature_folders = ['ml']

	# create features for each feature folder
	for f in feature_folders:

		# adjust feature location
		target_feature_location = os.path.join(feature_location, f)
		logging.debug('Loading features from: {}'.format(target_feature_location))
		# adjust model save location
		target_model_save_location = os.path.join(model_save_location, f)
		logging.debug('Saving models to: {}'.format(target_model_save_location))

		# get features from files
		X, Y, _ = extract_features_and_labels_from_files(target_feature_location)

		# define the grid_setup
		grid_setup = get_grid_setup()

		# define the pipeline setup
		pipeline_setup = get_pipeline_setup()

		# create and save the model
		execute_gridsearch_cv(X, Y, test_size = .2, shuffle = True, pipeline_setup = pipeline_setup, grid_setup = grid_setup, cv = 10, n_jobs = cpu_count(), scoring = 'f1_weighted', verbose = 10, save_model = True, model_save_location = target_model_save_location)

def calculate_ci_ml_classifier(models_folder = os.path.join('files', 'ml-models-nowac1', 'ml-32-nofilter-minus-round')):

	# read folder and only take models
	F = [ f for f in read_directory(models_folder) if f[-4:] == '.csv']

	# total number of test samples
	test_samples = 912

	# store all results
	all_results = pd.DataFrame()

	# loop over each file
	for f in F:

		logging.info(f'Processing file: {f}')

		# extract classification type from file name
		classification_type = f.split(os.sep)[-1][:-4]

		# add empty dictionary to dictionary
		results = {}

		# read csv as dataframe
		data = pd.read_csv(f, index_col = 0)

		# metrics
		metrics = ['test_precision', 'test_recall', 'test_f1']

		for m in metrics:

			metric_value = float(data.loc[m][0])

			results[m] = 1.96 * sqrt( (metric_value * (1 - metric_value)) / test_samples)
			
		all_results[classification_type] = pd.Series(results)
	
	# save to file
	all_results.to_csv(os.path.join(models_folder, '95_CI.csv'))
	

def create_dl_classifier(feature_location = os.path.join('labels', 'actigraph-actiwave-mapping', 'non-wear-time-features', 'dl-1-nofilter'), model_save_location = os.path.join('files', 'dl-models')):
	"""
	Create deep learning classifier

	Parameters
	----------
	feature_location: os.path
		location with the labeled features
	model_save_location: os.path
		folder where the ML models need to be saved
	"""

	# read directory with numpy arrays
	F_x = sorted([f for f in read_directory(feature_location) if f[-6:-4] == '_X'])
	F_y = sorted([f for f in read_directory(feature_location) if f[-6:-4] == '_Y'])

	# combine the two list of tuples so x an y matches
	F = list(zip(F_x, F_y))

	# small check to see if tuple has matching subject names
	random_index_checker = get_random_number_between(0, len(F))
	if F[random_index_checker][0][:-6] != F[random_index_checker][1][:-6]:
		logging.error('Sorting of tuples not done correctly.')
		exit(1)

	# create empty X and Y lists that can be populated with x and y features later
	X = []
	Y = []

	# loop over each tuple in F, which contains both the x and y location of features/labels, and add to X, Y list
	for file_x, file_y in F:

		# load numpy array from file
		x = np.load(file_x)
		y = np.load(file_y)

		# add to X and Y list
		X.append(x)
		Y.append(y)

	# vertical stack individual numpy arrays in list to a single numpy array
	X = np.vstack(X)
	Y = np.vstack(Y)

	logging.debug('Final X has shape: {}'.format(X.shape))
	logging.debug('Final Y has shape: {}'.format(Y.shape))

	# train multi-layer-perceptron with X and Y
	train_mlp_classifier(X,Y)


def explore_misclassification_of_training_data(feature_location = os.path.join('labels', 'actigraph-actiwave-mapping', 'non-wear-time-features', 'ml-32-nofilter-minus-round'), ml_classifier = os.path.join('files', 'ml-models-nowac1', 'ml-32-nofilter-minus-round', 'SVC.pkl')):
	"""
	Load labeled training features and explore the false positives and negatives. This will provide subject ID + ID of the candidate segments that cannot be classified into the correct class, thus wear or non-wear time

	Parameters
	---------
	feature_location : os.path
		folder location where features are stored
	ml_classifier : os.path
		location of trained machine learning classifier saved as pickle object
	"""
	
	# load machine learning classifier
	classifier = load_pickle(ml_classifier)

	# create empty dataframe
	df = pd.DataFrame()

	# loop over each file, open it, and add the data to the dataframe
	for f in [f for f in read_directory(feature_location) if f[-4:] == '.pkl']:

		# open .pkl file (this was pickled as a pandas dataframe)
		df = pd.concat([df, load_pickle(f)], axis = 1, sort=False)

	# transpose the datframe
	df = df.T

	# loop over each row of the dataframe
	for index, row in df.iterrows():

		# extract the label from the row
		y = np.array(row['label'], dtype = np.int)

		# get the duration of the segment
		duration = np.asarray(row['stop'] - row['start'], dtype='timedelta64[m]')
		
		# delete some collums that are not necessary for the features
		del row['start'], row['stop'], row['counter'], row['label']

		# get the X features
		X = np.array(row.values, dtype= np.float).reshape(1,-1)

		# predict the label
		y_hat = predict_class(classifier, X)[0]

		# check if y_hat is the same as the true y label
		if not np.equal(y,y_hat):
			logging.info('ID: {}'.format(index))
			logging.info('label: {}, predicted: {}'.format(y, y_hat))
			logging.info('duration: {}\n'.format(str(duration)))


def batch_process_detect_true_non_wear_time(limit = None, skip_n = 0, use_parallel = False, num_jobs = cpu_count(), ml_classifier = os.path.join('files', 'ml-models-nowac1', 'ml-32-nofilter-minus-round', 'SVC.pkl')):
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
	ml_classifier : os.path
		location of trained machine learning classifier saved as pickle object
	"""

	# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
	subjects = get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)[0 + skip_n:limit]

	# exclude subjects that have issues with their data
	subjects = [s for s in subjects if s not in get_subjects_with_invalid_data()]

	# # TESTING
	# subjects = ['90015722']
	# subjects = ['90021214']

	# load machine learning classifier
	classifier = load_pickle(ml_classifier)

	# loop over the subjects
	if use_parallel:

		# verbose
		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(process_detect_true_non_wear_time)(subject = subject, classifier = classifier, idx = i, total = len(subjects)) for i, subject in enumerate(subjects))
		# execute task
		executor(tasks)

	else:

		# verbose
		logging.info('Processing one-by-one (parallelization off)')

		# loop over the subjects
		for i, subject in enumerate(subjects):

			process_detect_true_non_wear_time(subject = subject, classifier = classifier, idx = i, total = len(subjects))


def process_detect_true_non_wear_time(subject, classifier, idx = 1, total = 1, perform_autocalibration = False, perform_correction = True, save_annotations_folder = os.path.join('labels', 'start_stop_all')):
	"""
	Detect non-wear time in actigraph acceleration data by including actiwave acceleration data
	basically when actigraph shows no activity, but actiwave records activity, there is a mismatch between the two signals
	this is the assumed non-wear time

	- find candidate non-wear segments in actigraph
	- map this with segment in actiwave (based on time)
	- calculate similarity between two signals (comparing VMU) with e.g. euclidian distance or dynamic time warping
	- find two groups within the set of similarity scores, for instance, by using gaussian mixture models
	- one group is the non-wear time, and the other group is the wear time

	Parameters
	---------
	subject : string
		subject ID
	classifier : sklearn classifier
		trained machine learning classifier to predict wear or non-wear time from acceleration data
	idx : int (optional)
		index tracker for progress
	total : int (optional)
		total number of files to process, used to monitor status
	perform_autocalibration : Boolean (optional)
		if set to True, apply autocalibration of acceleration Y, X, and Z axes
	perform_correction : Boolean (optional)
		if set to True, subjects with long candidate non wear sequence that typically gets misclassified as wear time will be corrected
	"""

	# verbose
	logging.info('{style} Processing subject: {} {}/{} {style}'.format(subject, idx, total, style = '='*10))

	"""
		GET ACTIGRAPH AND ACTIWAVE DATA
	"""

	# actigraph acceleration data
	actigraph_acc, actigraph_meta_data, actigraph_time = get_actigraph_acc_data(subject = subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE, autocalibrate = perform_autocalibration)
	# actiwave acceleration data
	actiwave_acc, _, actiwave_time = get_actiwave_acc_data(subject = subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE, autocalibrate = perform_autocalibration)
	# actiwave heart rate data
	actiwave_hr, actiwave_hr_time = get_actiwave_hr_data(subject = subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)


	"""
		FIND CANDIDATE NON-WEAR SEGMENTS ACTIGRAPH ACCELERATION DATA
	"""

	# for actigraph
	actigraph_non_wear_vector = find_candidate_non_wear_segments_from_raw(actigraph_acc, std_threshold = 0.004, min_segment_length = 1, sliding_window = 1, hz = int(actigraph_meta_data['Sample_Rate']))

	"""
		GET START AND END TIME OF NON WEAR SEGMENTS
	"""

	# find all indexes of the numpy array that have been labeled non-wear time
	non_wear_indexes = np.where(actigraph_non_wear_vector == 0)[0]
	# find consecutive ranges
	non_wear_segments = find_consecutive_index_ranges(non_wear_indexes)
	# empty dictionary where we can store the start and stop times
	dic_segments = {}
	
	# check if segments are found
	if len(non_wear_segments[0]) > 0:
		
		# find start and stop times (the indexes of the edges and find corresponding time)
		for i, row in enumerate(non_wear_segments):

			# find start and stop
			start, stop = np.min(row), np.max(row)

			# add the start and stop times to the dictionary
			dic_segments[i] = {'start': actigraph_time[start], 'stop' : actigraph_time[stop], 'start_index': start, 'stop_index' : stop}

	"""
		RESAMPLE DATA
	"""

	# resample 100hz actigraph acceleration data to 32 hertz. 
	actigraph_acc = resample_acceleration(actigraph_acc, 100, 32, False)

	# check if both arrays have equal lengths
	if actigraph_acc.shape[0] != actiwave_acc.shape[0]:
		
		logging.warning('Actigraph and Actiwave signals have unequal number of samples. {} vs {}. Clipping on the lowest number'.format(actigraph_acc.shape[0], actiwave_acc.shape[0]))

		# get the clip length
		min_length = min(actigraph_acc.shape[0], actiwave_acc.shape[0])
		# clip actigraph acceleration
		actigraph_acc = actigraph_acc[: min_length]
		# clip actiwave acceleration 
		actiwave_acc = actiwave_acc[:min_length]
		# and clip the time array
		actiwave_time = actiwave_time[:min_length]

	"""
		ADD VMU TO DATA
	"""

	# create actigraph VMU data
	actigraph_vmu = calculate_vector_magnitude(actigraph_acc, minus_one = True, round_negative_to_zero = True)
	# add VMU to actigraph acceleration
	actigraph_acc = np.hstack((actigraph_acc, actigraph_vmu))
	# create actigraph VMU data
	actiwave_vmu = calculate_vector_magnitude(actiwave_acc, minus_one = True, round_negative_to_zero = True)
	# add VMU to actigraph acceleration
	actiwave_acc = np.hstack((actiwave_acc, actiwave_vmu))

	"""
		CREATING THE DATAFRAMES
	"""

	# create dataframe of actigraph acceleration data, inclusing VMU data. Note that we use the actiwave time data since we have resamples to data from 100hz to 32hz, which is the sampling frequency of actiwave.
	df_actigraph_acc = pd.DataFrame(actigraph_acc, index = actiwave_time, columns = ['Y', 'X', 'Z', 'VMU ACTIGRAPH'])
	# create dataframe of actiwave acceleration + VMU
	df_actiwave_acc = pd.DataFrame(actiwave_acc, index = actiwave_time, columns = ['Y', 'X', 'Z', 'VMU ACTIWAVE'])
	# create dataframe of actiwave acceleration
	df_actiwave_hr = pd.DataFrame(actiwave_hr, index = actiwave_hr_time, columns = ['HR ACTIWAVE'])


	"""
		FIND NON WEAR TIME
	"""

	# new non wear vector that contains the true non-wear time after scoring 
	non_wear_final_vector = np.ones((actigraph_time.shape[0], 1))

	# empty list for start and stop data
	annotations = pd.DataFrame()

	# only find scores when there are segments found
	if len(dic_segments) > 0:

		# loop over the dictionary start and stop times
		counter = 0
		for v in dic_segments.values():

			# get the start and stop values from the value componont
			start, stop = v['start'], v['stop']
			logging.debug('Processing Start: {}, Stop : {}, Counter : {}'.format(start, stop, counter))

			# obtain correct slice for actigraph data
			df_actigraph_segment = df_actigraph_acc.loc[start:stop]
			# obtain correct slice for actigraph data
			df_actiwave_segment = df_actiwave_acc.loc[start:stop]
			# obtain correct slice for actigraph heart rate data
			df_actiwave_hr_segment = df_actiwave_hr.loc[start:stop]

			# join the acceleration data
			joined_segment = df_actigraph_segment.iloc[:,[3]].join(df_actiwave_segment.iloc[:,[3]], how = 'inner')
			# add heart rate data
			joined_segment = joined_segment.join(df_actiwave_hr_segment.iloc[:,[0]], how = 'inner')
			
			# give easy variable name to column and drop NaNs
			a = joined_segment['VMU ACTIGRAPH'].dropna().values
			b = joined_segment['VMU ACTIWAVE'].dropna().values
			c = joined_segment['HR ACTIWAVE'].dropna().values
			
			# get features for candidate segment
			df_features = pd.DataFrame(get_features_for_candidate_segment(a, b, c, None))

			# convert to numpy array
			X = np.array(df_features.values, dtype = np.float).reshape(1,-1)

			# predict class
			label = predict_class(classifier, X)[0]

			# if perform_correction is set to True, correct label for subjects with long candidate non wear sequence; typically this long sequence will be classified as wear time where it should be non wear time
			if perform_correction:
				
				# check if subject is part of subjects for correction dictionary, if so, check if counter is part of returned list
				counter_to_correct = get_subject_counters_for_correction(subject)
				if counter_to_correct is not None:
					if counter in counter_to_correct:
						# correct label by flipping it
						label = 0 if label == 1 else 1
			
			logging.debug('Predicted class label: {}'.format(label))

			# check if label is predicted as non-wear time, that is 0
			if label == 0:
				non_wear_final_vector[v['start_index']:v['stop_index']] = 0

			annotations[counter] = pd.Series({'counter' : counter, 'start' : v['start'], 'stop' : v['stop'], 'start_index': v['start_index'], 'stop_index' : v['stop_index'], 'label' : label})
			counter += 1

	# create folder to save annotations to
	create_directory(save_annotations_folder)

	# save annotations
	annotations.T.to_csv(os.path.join(save_annotations_folder, f'{subject}.csv'))

	# plot_non_wear_data(actigraph_acc = df_actigraph_acc, actiwave_acc = df_actiwave_acc, actiwave_hr = df_actiwave_hr, plot_folder = os.path.join('plots','actiwave-nw-with-counter'), subject = subject, annotations = annotations.T)

	# save to HDF5 if set to True
	# save_data_to_group_hdf5(group = subject, data = non_wear_final_vector, data_name = 'actigraph_true_non_wear', overwrite = True, create_group_if_not_exists = False, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
		

def batch_process_calculate_classification_performance(function, limit = None, skip_n = 0, use_parallel = False, num_jobs = cpu_count()):
	"""
	Batch to calculate classification performance (precisions, recall, F1) for true non wear time compared to the existing non wear methods such as
	Troiano, Choi, Hees, Hecht.

	Parameters
	-----------
	function : python function
		function to call with batch processing
	limit : int (optional)
		limit the number of subjects to be processed
	skip_n : int (optional)
		skip first N subjects
	use_parallel = Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs = int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	"""

	# verbose
	logging.info('Start batch processing calculate classification performance')

	# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
	subjects = get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)[0 + skip_n:limit]

	# remove subjects with invalid data
	subjects = [s for s in subjects if s not in get_subjects_with_invalid_data()]

	# loop over the subjects
	if use_parallel:

		# verbose
		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(function)(subject = subject, idx = i, total = len(subjects)) for i, subject in enumerate(subjects))
		# execute task
		executor(tasks)

	else:

		# verbose
		logging.info('Processing one-by-one (parallelization off)')

		# loop over the subjects
		for i, subject in enumerate(subjects):

			function(subject = subject, idx = i, total = len(subjects))


def process_calculate_classification_performance(subject, idx = 1, total = 1):
	"""
	Process classification performance of true non wear time (detected using actiwave data) and estimated non wear time, calculated
	based on existing non wear methods (such as Choi, Troiano, Hees, Hecht).

	Parameters
	----------
	subject : string
		subject ID to process
	idx : int (optional)
		index tracker for progress
	total : int (optional)
		total number of subjects to process. Is used to monitor the progress
	"""

	# verbose
	logging.info('{style} Processing subject:{} {}/{} {style}'.format(subject, idx, total, style = '='*10))

	"""
		OBTAIN THE NON WEAR VECTORS
	"""

	# true non wear time
	true_nw = read_dataset_from_group(group_name = subject, dataset = 'actigraph_true_non_wear', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) 
	# hecht 3-axes non wear time
	hecht_3_nw = read_dataset_from_group(group_name = subject, dataset = 'hecht_2009_3_axes_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	# check if hecht is None, if so, there is no need to continue. This happens when there was no epoch data to begin with
	if hecht_3_nw is None:
		logging.warning('Dataset hecht_2009_3_axes_non_wear_data returned with None value. Skipping subject {}...'.format(subject))
		return
	else:
		# take all values in column index 1, column index 0 is the time array which we don't need
		hecht_3_nw = hecht_3_nw[:,1]

	# troiano non wear time, take all values in column index 1, column index 0 is the time array which we don't need
	troiano_nw = read_dataset_from_group(group_name = subject, dataset = 'troiano_2007_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)#[:,1]
	# get the start time from the first row of the first column
	# start_datetime = np.asarray(troiano_nw[0,0], dtype='datetime64[ns]')
	start_datetime = troiano_nw[0,0]

	# get the second column of troiano_nw since it contains the non-wear data
	troiano_nw = troiano_nw[:,1]
	# choi non wear time, take all values in column index 1, column index 0 is the time array which we don't need
	choi_nw = read_dataset_from_group(group_name = subject, dataset = 'choi_2011_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)[:,1]
	# hees non wear time
	hees_nw = read_dataset_from_group(group_name = subject, dataset = 'hees_2013_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

	"""
		UPSCALE AND DOWNSCALE SO TIME INTERVALS ARE ALL ON 1 SEC 
	"""	

	# true non wear time, go from 100 measurements per second to 1 per sec
	true_nw = true_nw[::100]
	# hecht 3 axes non wear time, go from 1 per 60s to 1 per sec
	hecht_3_nw = hecht_3_nw.repeat(60, axis = 0)
	# troiano  non wear time, go from 1 per 60s to 1 per sec
	troiano_nw = troiano_nw.repeat(60, axis = 0)
	# choi non wear time, go from 1 per 60s to 1 per sec
	choi_nw = choi_nw.repeat(60, axis = 0)
	# hees non wear time, go from 100 measurements per second to 1 per sec
	hees_nw  = hees_nw[::100]

	# clip all non wear time vectors based on the shortest one
	clip_length = min(len(true_nw), len(hecht_3_nw), len(troiano_nw), len(choi_nw), len(hees_nw))

	# clip all non wear time vectors, and swap 1->0 and 0->1 and reshape to have n_samples x 1 column
	true_nw = 1 - true_nw[:clip_length].reshape(-1,1)
	hecht_3_nw = 1 - hecht_3_nw[:clip_length].reshape(-1,1)
	troiano_nw = 1 - troiano_nw[:clip_length].reshape(-1,1)
	choi_nw = 1 - choi_nw[:clip_length].reshape(-1,1)
	hees_nw = 1 - hees_nw[:clip_length].reshape(-1,1)

	# combine data
	data = np.hstack([true_nw, hecht_3_nw, troiano_nw, choi_nw, hees_nw]).astype(np.uint8)

	# add start time to meta_data; this can later be used to create a time column
	meta_data = {'start_datetime': start_datetime}

	save_data_to_group_hdf5(group = subject, data = data, data_name = 'non_wear_data', meta_data = meta_data, overwrite = True, create_group_if_not_exists = True, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)


def process_save_classification_performance_to_file(save_folder = os.path.join('files', 'classification-performance')):
	"""
	Read classification performance for each subject and save as a CSV file to disk. This enables faster post-processing of this data
	In addition, save a combined numpy array with all the classification results; true labels and the four non wear algorithms

	Parameters
	----------
	df_save_folder: os.path
		folder location to save data to	
	"""

	# define start and stop time for filtered data; here we can define to filter data for day time only
	filter_start, filter_end = '07:00', '23:00'

	# get all the subjects from the hdf5 file and remove subjects with invalid data
	subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]

	# create save folder if not exist
	create_directory(save_folder)

	# empty list to store the non wear data to
	non_wear_data = []
	non_wear_data_filtered = []

	# loop over each subject and read performance metrics from hdf5
	for i, subject in enumerate(subjects):

		# verbose
		logging.info('{style} Processing subject:{} {}/{} {style}'.format(subject, i, len(subjects), style = '='*10))

		"""
			READ DATA
		"""

		# read data from HDF5
		data = read_dataset_from_group(group_name = subject, dataset = 'non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
		
		# check if data is not None
		if data is None:
			logging.warning('Dataset returned with None value. Skipping subject {}...'.format(subject))
			continue

		# add data to non wear data list
		non_wear_data.append(data)
		
		# get meta data with classification statistics
		meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

		"""
			FILTER DATA
		"""

		# read start_datetime and remove from dictionary at the same time and convert to datetime64
		start_datetime = np.asarray(meta_data.pop('start_datetime'), dtype='datetime64[ns]')
		
		# convert start_datetime into an array with timestamps per second with the length being equal to the lenght of the data
		time_data = np.asarray(start_datetime, dtype='datetime64[s]') + np.asarray(np.arange(data.shape[0]), dtype='timedelta64[s]')

		# create pandas dataframe from data and time
		df_data = pd.DataFrame(data, index = time_data)
		
		# filter between start and stop and convert to numpy array
		data_filtered = df_data.between_time(filter_start, filter_end).to_numpy()
		
		# add filtered data to list
		non_wear_data_filtered.append(data_filtered)

	# convert list of arrays to single numpy array
	non_wear_data = np.vstack((non_wear_data))
	non_wear_data_filtered = np.vstack((non_wear_data_filtered))

	# save numpy data to file
	np.save(os.path.join(save_folder, 'non_wear_data.npy'), non_wear_data)
	np.save(os.path.join(save_folder, 'non_wear_data_filtered.npy'), non_wear_data_filtered)



def process_classification_performance_total(data_folder = os.path.join('files', 'classification-performance')):
	"""
	Create table with classification performance of all subjects

	Paramaters
	----------
	data_folder : os.path
		folder where the classification numpy array is stored
	"""

	# files to process
	F = ['non_wear_data', 'non_wear_data_filtered']

	# loop over files and process
	for f in F:

		logging.info('Processing file: {}'.format(f))

		# read non wear data
		non_wear_data = np.load(os.path.join(data_folder, '{}.npy'.format(f)))

		# define the labels of the array columns
		col_to_label = {0: 'True', 1 : 'Hecht', 2 : 'Troiano', 3 : 'Choi', 4 : 'Hees'}

		# dataframe to store values to
		df = pd.DataFrame()

		# loop over data from non wear algorithm and calculate classification performance
		for col in range(1, non_wear_data.shape[1]):

			logging.debug('Processing column {} {}'.format(col, col_to_label[col]))

			# true labels; this is the first column, or index 0
			y = non_wear_data[:,0]

			# get predicted values
			y_hat = non_wear_data[:,col]

			# get confusion matrix values
			tn, fp, fn, tp = get_confusion_matrix(y, y_hat, labels = [0,1]).ravel()

			logging.debug('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

			# calculate classification performance such as precision, recall, f1 etc.
			classification_performance = calculate_classification_performance(tn, fp, fn, tp)

			# add classification performance to dataframe
			df[col_to_label[col]] = pd.Series(classification_performance)
		
		# transpose dataframe
		df = df.T

		df.to_csv(os.path.join(data_folder, 'classification-results-{}.csv'.format(f)))


def process_plot_classification_results(data_folder = os.path.join('files', 'classification-performance')):
	"""
	plot classification performance of all subjects

	Paramaters
	----------
	data_folder : os.path
		folder where the classification numpy array is stored
	"""

	# files to process
	F = ['non_wear_data', 'non_wear_data_filtered']

	# read in csv file as dataframe
	data = pd.read_csv(os.path.join(data_folder, 'classification-results-{}.csv'.format(F[0])), index_col = 0)
	data_filtered = pd.read_csv(os.path.join(data_folder, 'classification-results-{}.csv'.format(F[1])), index_col = 0)

	"""
		Calculate 95% Confidence interval
	"""
	data_ci = pd.DataFrame()

	for index, rows in data.iteritems():

		if index in ['accuracy', 'precision', 'recall', 'f1']:

			row_data = {}

			for row_index, row in rows.iteritems():

				row = float(row)

				# update dataframe
				row_data[row_index] = 1.96 * sqrt( (row * (1 - row)) / 583)

			data_ci[index] = pd.Series(row_data)

	"""
		Calculate 95% Confidence interval
	"""
	data_filtered_ci = pd.DataFrame()

	for index, rows in data_filtered.iteritems():

		if index in ['accuracy', 'precision', 'recall', 'f1']:

			row_data = {}

			for row_index, row in rows.iteritems():

				row = float(row)

				# update dataframe
				row_data[row_index] = 1.96 * sqrt( (row * (1 - row)) / 583)

			data_filtered_ci[index] = pd.Series(row_data)


	# call plot function
	plot_classification_results(data, data_ci, data_filtered, data_filtered_ci)


def process_plot_time_distribution(load_from_file = True):

	if not load_from_file:
		# read subjects
		subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]

		# get a list with only hours, value is [all, < 1 hour, > 1 hour]
		hours = {str(x.hour).zfill(2) : [0,0,0] for x in pd.date_range('2017-01-01 00:00', '2017-01-01 23:59', freq = '60T')}
		
		# loop over each subject, get the time distribution
		for i, subject in enumerate(subjects):

			logging.info('{style} Processing subject : {} {}/{} {style}'.format(subject, i, len(subjects), style = '='*10))

			*_, actigraph_time = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

			# true non wear time
			true_non_wear_time = 1 - read_dataset_from_group(group_name = subject, dataset = 'actigraph_true_non_wear', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

			# downscale array for faster processing
			true_non_wear_time = true_non_wear_time[::100*60]
			actigraph_time = actigraph_time[::100*60]

			# create dataframe of true non wear time
			df = pd.DataFrame(true_non_wear_time, index = actigraph_time, columns = ['nw'])
			
			for _, group in df.groupby(df.index.hour).nw:

				# get the hour
				group_hour = group.index.strftime('%H')[0]

				# if group is larger than zero, then some non-wear is identified
				if group.sum() > 0:
					
					# add one to hours dictionary
					hours[group_hour][0] +=1
					
					# add 
					if group.sum() <= 59:
						hours[group_hour][1] +=1
					else:
						hours[group_hour][2] +=1
		
		# save hours to temp file
		save_pickle(hours, 'hours', 'files')
	else:
		hours = load_pickle('hours', 'files')

	plot_time_distribution(hours)

def process_plot_nw_scenarios(read_from_file = True):

	if not read_from_file:
		# first scenario: non wear time
		data = {'90097934' : {'start' : '10:00', 'stop' : '12:00'},
				'90079227' : {'start' : '11:00', 'stop' : '12:00'},
				'90222419' : {'start' : '10:00', 'stop' : '11:00'},
					}

		# obtain data for each subject
		for subject in data.keys():

			logging.info('Processing subject {}'.format(subject))

			"""
				GET ACTIGRAPH DATA
			"""
			actigraph_acc, _ , actigraph_time = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
			# get start and stop time
			# start_time, stop_time = actigraph_time[0], actigraph_time[-1]

			# obtain candidate segments
			candidate_segment = find_candidate_non_wear_segments_from_raw(actigraph_acc, std_threshold = 0.004, min_segment_length = 1, sliding_window = 1, hz = 100)

			"""
				GET ACTIWAVE DATA
			"""
			actiwave_acc, _ , actiwave_time = get_actiwave_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
			actiwave_hr, actiwave_hr_time = get_actiwave_hr_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

			"""
				CREATING THE DATAFRAMES
			"""

			# convert actigraph data to pandas dataframe
			df_actigraph_acc = pd.DataFrame(actigraph_acc, index = actigraph_time, columns = ['ACTIGRAPH Y', 'ACTIGRAPH X', 'ACTIGRAPH Z'])
			# candidate segment
			df_candidate_segment = pd.DataFrame(candidate_segment, index = actigraph_time, columns = ['CANDIDATE NW EPISODE'])
			# convert actiwave data to pandas dataframe
			df_actiwave_acc = pd.DataFrame(actiwave_acc, index = actiwave_time, columns = ['ACTIWAVE Y', 'ACTIWAVE X', 'ACTIWAVE Z'])
			# convert actiwave hr to pandas dataframe
			df_actiwave_hr = pd.DataFrame(actiwave_hr, index = actiwave_hr_time, columns = ['ESTIMATED HR'])

			"""
				COMBINE DATAFRAMES
			"""
			# merge all dataframes
			df_data = df_actigraph_acc \
						.join(df_candidate_segment, how='outer') \
						.join(df_actiwave_acc, how='outer') \
						.join(df_actiwave_hr, how='outer')

			# cut additional day for specific subject
			if subject == '90222419':
				df_data = df_data.iloc[:int(len(df_data) * 0.5)]

			# filter between start and stop
			df_data = df_data.between_time(data[subject]['start'], data[subject]['stop'])

			data[subject]['data'] = df_data
	else:
		# save data as pickle
		data = load_pickle('nw_scenarios', os.path.join('files', 'paper'))

	plot_nw_scenarios(data)

def process_table_cv_classification(grid_search_folder = os.path.join('files', 'grid-search', 'final_reverse_prec_rec_new_hecht'), cv_grid_search_folder = os.path.join('files', 'grid-search-cv', 'split3-false_reverse_new_hecht')):

	classification_metrics = ['accuracy', 'precision', 'recall', 'f1']

	# read grid search files from folder
	gs_files = read_directory(grid_search_folder)
	# read cross validation grid search filres
	cv_gs_files = [x for x in read_directory(cv_grid_search_folder) if 'cv-grid-search-results' in x]

	# empty dataframe to store classification results to
	df_classification = pd.DataFrame()
	# empty dataframe for parameter combinations
	df_parameters = pd.DataFrame(index = ['AT', 'MPL', 'ST', 'SS', 'VM', 'WST', 'MWL', 'T', 'I', 'M', 'MW', 'WO', 'StdT', 'StdA', 'VT', 'VA'])

	"""
		CV GRID SEARCH
	"""
	for f in cv_gs_files:

		# read pkl file
		data = load_pickle(f)

		# read method name from file
		nw_method = f.split('-')[-1][:-4]

		df_training_results = pd.DataFrame()
		for cnt, training_results in data['training'].items():
			df_training_results[cnt] = pd.Series(training_results['test_results'])

		# add training classification results as a series to the dataframe
		df_classification['{} {}'.format(nw_method, 'training')] = df_training_results.T.mean(skipna = True)
		# add test classification results as a series to the dataframe
		df_classification['{} {}'.format(nw_method, 'test')] = pd.Series(data['test'])

		# get parameters that resulted in highest classification result
		if nw_method == 'troiano':
			# unpack cross validation parameter values
			at, mpl, st, ss, vm  = data['combination'].split('-')
			# add to dataframe
			df_parameters['{} {}'.format(nw_method, 'f1 (cv)')] = pd.Series({'AT' : int(at), 'MPL' : int(mpl), 'ST' : int(st), 'SS' : int(ss), 'VM' : vm})
			# also add the default parameter values (see non wear algorithm for default values)
			df_parameters['{} {}'.format(nw_method, 'default')] = pd.Series({	'AT' : 0, 'MPL' : 60, 'ST' : 2, 'SS' : 100, 'VM' : 'False'})
		elif nw_method == 'choi':
			# same as above
			at, mpl, st, mwl, wst, vm = data['combination'].split('-')
			df_parameters['{} {}'.format(nw_method, 'f1 (cv)')] = pd.Series({'AT' : int(at), 'MPL' : int(mpl), 'ST' : int(st), 'MWL' : int(mwl), 'WST' : int(wst), 'VM' : vm})
			df_parameters['{} {}'.format(nw_method, 'default')] = pd.Series({	'AT' : 0, 'MPL' : 90, 'ST' : 2, 'MWL' : 30, 'WST' : 0, 'VM' : 'False'})
		elif nw_method == 'hecht':
			# same as above
			t, i, m = data['combination'].split('-')
			df_parameters['{} {}'.format(nw_method, 'f1 (cv)')] = pd.Series({'T' : int(t), 'I' : int(i), 'M' : int(m)})
			df_parameters['{} {}'.format(nw_method, 'default')] = pd.Series({	'T' : 5, 'I' : 20, 'M' : 2})
		elif nw_method == 'hees':
			mw, wo, st, sa, vt, va = data['combination'].split('-')
			df_parameters['{} {}'.format(nw_method, 'f1 (cv)')] = pd.Series({'MW' : int(mw), 'WO' : int(wo), 'StdT' : int(st), 'StdA' : int(sa), 'VT' : int(vt), 'VA' : int(va)})
			df_parameters['{} {}'.format(nw_method, 'default')] = pd.Series({'MW' : 60, 'WO' : 15, 'StdT' : 3, 'StdA' : 2, 'VT' : 50, 'VA' : 2})
		else:
			logging.error('Non wear method {} not implemented'.format(nw_method))
			exit(1)

	"""
		GRID SEARCH (on all the subjects)
	"""
	for f in gs_files:

		# read pkl file
		data = load_pickle(f)

		# read method name from file
		nw_method = f.split('-')[-1][:-4]

		# # print default parameter classification results (for test)
		# if nw_method == 'hecht':
		# 	logging.info('Default Hecht classification results: {}'.format(data['5-20-2']))
		# elif nw_method == 'troiano':
		# 	logging.info('Default Troiano classification results: {}'.format(data['0-60-2-100-False']))
		# elif nw_method == 'hees':
		# 	logging.info('Default Hees classification results: {}'.format(data['60-15-3-2-50-2']))
		# elif nw_method == 'choi':
		# 	logging.info('Default Choi classification results: {}'.format(data['0-90-2-30-0-False']))

		for classification_metric in classification_metrics:

			# variables to keep track of the top classification results and the classification metric
			top_results, top_value = [], 0

			for row in sorted(data.items(), key = lambda item: item[1][classification_metric], reverse = True):

				# only do this for the first result (the top result)
				if len(top_results) == 0:
					# add row to top results
					top_results.append(row)
					top_value = row[1][classification_metric]

					# add classification results to dataframe
					df_classification['{} {}'.format(nw_method, 'all')] = pd.Series(row[1])
				else:
					# now check if second and other results have similar score as the first one (sometimes multiple combinatations result in the same classification score)
					if row[1][classification_metric] == top_value:
						top_results.append(row)
					else:
						break
			
			# temporary dataframe to store parameter values to
			df_temp = pd.DataFrame()
			# get parameter values and store in dataframe
			for i, row in enumerate(top_results):
				if nw_method == 'choi':
					at, mpl, st, mwl, wst, vm = row[0].split('-')
					df_temp[i] = pd.Series({'AT' : int(at), 'MPL' : int(mpl), 'ST' : int(st), 'MWL' : int(mwl), 'WST' : int(wst), 'VM' : vm})
				elif nw_method == 'troiano':
					at, mpl, st, ss, vm  = row[0].split('-')
					df_temp[i] = pd.Series({'AT' : int(at), 'MPL' : int(mpl), 'ST' : int(st), 'SS' : int(ss), 'VM' : vm})
				elif nw_method == 'hecht':
					t, i, m = row[0].split('-')
					df_temp[i] = pd.Series({'T' : int(t), 'I' : int(i), 'M' : int(m)})
				elif nw_method == 'hees':
					mw, wo, st, sa, vt, va = row[0].split('-')
					df_temp[i] = pd.Series({'MW' : int(mw), 'WO' : int(wo), 'StdT' : int(st), 'StdA' : int(sa), 'VT' : int(vt), 'VA' : int(va)})
				else:
					logging.error('Non wear method top results not implemented: {}'.format(nw_method))
					exit()
			

			# get ranges of parameter values (for instance, 10, 20, 30 needs to be converted to 10-30)
			temp_data = {}
			for name, values in df_temp.iterrows():
				# get min and max parameter value
				min_value, max_value = values.min(), values.max()
				
				# add as combination, for example 10-30 if min and max are not the same, and if they are the same, then just use min (this avoids ranges such as 10-10)
				if min_value != max_value:
					temp_data[name] = '{}--{}'.format(min_value, max_value)
				else:
					temp_data[name] = '{}'.format(min_value)
			
			# add to dataframe 
			df_parameters['{} {}'.format(nw_method, classification_metric)] = pd.Series(temp_data)

	
	# transpose so we have access to colums
	df_parameters = df_parameters.T

	# merge key values that mean the same thing
	for from_key, to_key in [('MPL', 'MW'), ('WST', 'M'), ('MWL', 'I')]:
		
		# combine MPL and MW (they both minimum interval (mins))
		combine_mpl_mw = df_parameters[from_key].astype(str).combine(df_parameters[to_key].astype(str), min)
		# replace column in dataframe
		df_parameters[to_key] = combine_mpl_mw
		# remove mpl
		df_parameters = df_parameters.drop(from_key, axis=1)
	

	# transpose back again
	df_parameters = df_parameters.T

	# add column with short code to long code conversion
	df_parameters['description'] = df_parameters.index
	df_parameters['description'] = df_parameters['description'].apply(convert_short_code_to_long)

	# remove columns
	for index_row in ['tn', 'fp', 'fn', 'tp']:
		df_classification = df_classification.drop(index_row)


	# pd.options.display.float_format = '{:.3f}'.format
	# print(df_classification.round(3))
	df_classification = df_classification.round(3)
	# sort
	df_classification = df_classification.reindex(sorted(df_classification), axis = 1)
	df_parameters = df_parameters.fillna('-')
	df_parameters = df_parameters.replace('nan', '-', regex=True)
	
	# sort columns
	sorted_columns = []
	for column in df_parameters:
		nw_method_sort = {'hecht' : 0, 'troiano' : 1, 'choi' : 2, 'hees' : 3}
		metric_sort = {'default' : 0, 'accuracy' : 1, 'precision' : 2, 'recall' : 3, 'f1' : 4, 'f1 (cv)' : 5}

		substring_nw_method = column.split()[0]
		substring_metric = ' '.join(column.split()[1:])

		column = '{}{}{}'.format(nw_method_sort.get(substring_nw_method, ''), metric_sort.get(substring_metric, ''), column)

		if column == 'description': column = '00description'
		sorted_columns.append(column)
	sorted_columns = [x[2:] for x in sorted(sorted_columns)]
	df_parameters = df_parameters.reindex(sorted_columns, axis=1)

	"""
		Calculate 95% Confidence interval
	"""
	df_classification_with_ci = pd.DataFrame()

	for index, rows in df_classification.iteritems():

		total = 583

		if 'all' in index:
			n = total
		elif 'test' in index:
			n = int(.3 * total)
		elif 'training' in index:
			n = int(.7 * total)
		else:
			logging.error('cannot calculate number of samples')
			exit(1)

		data = {}

		for row_index, row in rows.iteritems():

			row = float(row)

			ci = 1.96 * sqrt( (row * (1 - row)) / n)

			# update dataframe
			data[row_index] = '{:.3f} $\pm$ {:.3f}'.format(round(row,3), round(ci,3))

		df_classification_with_ci[index] = pd.Series(data)


	# save to disk
	# df_classification.to_csv(os.path.join('plots', 'paper1', 'tables', 'cv_classification_results.csv'))
	df_classification_with_ci.to_csv(os.path.join('plots', 'paper1', 'tables', 'cv_classification_results_with_ci.csv'))
	# df_parameters.to_csv(os.path.join('plots', 'paper1', 'tables', 'parameter_values.csv'))


def process_nw_time_overview(read_from_file = True):
	"""
	Get an overview of the non-wear sequences in the data so as to know their distribution
	"""

	# get all the subjects from the hdf5 file and remove subjects with invalid data
	subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]

	# empty dictionary where we can store the lengths of the non wear time
	dic_segments = {}

	# keep track of the total duration
	total_data = []
		
	# loop over each subject, get the time distribution
	for i, subject in enumerate(subjects):

		logging.info('{style} Processing subject : {} {}/{} {style}'.format(subject, i, len(subjects), style = '='*10))

		# read true non wear time and convert 0>1 and 1->0
		subject_true_nw = 1 - read_dataset_from_group(group_name = subject, dataset = 'actigraph_true_non_wear', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE).astype('uint8').reshape(-1,1)
		# convert to 1s instead of 100hz
		subject_true_nw = subject_true_nw[::100]

		# add length in minutes of data to total data
		total_data.append(len(subject_true_nw) // 60)

		"""
			GET LENGTH NON WEAR SEGMENTS
		"""

		# find all indexes of the numpy array that have been labeled non-wear time
		non_wear_indexes = np.where(subject_true_nw == 1)[0]
		# find consecutive ranges
		non_wear_segments = find_consecutive_index_ranges(non_wear_indexes)
		
		# check if segments are found
		if len(non_wear_segments[0]) > 0:

			# get lenghts of segments and add to dictionary
			dic_segments[subject] = [len(x) for x in non_wear_segments]
	

	# get a list of lenghts in minutes
	data = []
	# get all non wear time lengths
	for value in dic_segments.values():
		for length in value:
			data.append(length // 60)

	# convert to numpy array
	data = np.array(data)
	logging.info('Total length of data: {}hrs'.format(np.sum(total_data) / 60))
	logging.info('Frequency of nw-time episodes: {}'.format(len(data)))
	logging.info('Mean of nw-time episodes: {}'.format(np.mean(data)))
	logging.info('SD of nw-time episodes: {}'.format(np.std(data)))
	logging.info('Total nw-time : {}hrs ({})'.format(np.sum(data) / 60, np.sum(data) / np.sum(total_data) * 100))

	# shorter than 60 minutes
	data_shorter = data[data < 60]
	logging.info('Shorter than 60 minutes:')
	logging.info('Frequency of nw-time episodes: {} {}'.format(len(data_shorter), len(data_shorter) / len(data) * 100))
	logging.info('Mean of nw-time episodes: {}'.format(np.mean(data_shorter)))
	logging.info('SD of nw-time episodes: {}'.format(np.std(data_shorter)))

	# 60 minutes or longer
	data_longer = data[data >= 60]
	logging.info('Longer or equal than 60 minutes:')
	logging.info('Frequency of nw-time episodes: {} {}'.format(len(data_longer), len(data_longer) / len(data) * 100))
	logging.info('Mean of nw-time episodes: {}'.format(np.mean(data_longer)))
	logging.info('SD of nw-time episodes: {}'.format(np.std(data_longer)))

	# plot data
	plot_nw_distribution(data)


def process_participants_characteristics():


	# get all the subjects from the hdf5 file and remove subjects with invalid data
	subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]

	# location of the CSV file with subject meta data
	meta_data_file = os.path.join(os.sep, 'Volumes', 'LaCie', 'AGD_TEST', 'vekt.csv')

	# convert CSV file to dictionary so we have access to each of the meta data by subject ID (skip the first row as it represents the column headers)
	meta_data = {x[4] : {'age' : x[0], 'sex' : 'Male' if x[1] == '1' else 'Female', 'height' : x[2], 'weight' : x[3]} for x in read_csv(meta_data_file)[1:]}

	data = pd.DataFrame(index = ['age', 'sex', 'height', 'weight'])

	for s in subjects:

		data[s] = pd.Series(meta_data[s])
	
	# transform dataframe
	data = data.T

	# change datatype of columns
	data[['age', 'height', 'weight']] = data[['age', 'height', 'weight']].astype('float')
	data['bmi'] = data['weight'] / (data['height']/100)**2
	
	# total number of participants
	total = 698
	errors = total - len(data) 
	n = len(data)
	n_male = sum(data['sex'] == 'Male')
	n_female = sum(data['sex'] == 'Female')
	perc_male = round(n_male / n * 100, 2)
	perc_female = round(n_female / n * 100, 2)
	min_age = int(data['age'].min())
	max_age = int(data['age'].max())
	avg_age = round(data['age'].mean(), 2)
	std_age = round(data['age'].std(), 2)
	avg_height = round(data['height'].mean(), 2)
	std_height = round(data['height'].std(), 2)
	avg_weight = round(data['weight'].mean(), 2)
	std_weight = round(data['weight'].std(), 2)
	# min_bmi = round(data['bmi'].min(), 2)
	# max_bmi = round(data['bmi'].max(), 2)
	avg_bmi = round(data['bmi'].mean(), 2)
	std_bmi = round(data['bmi'].std(), 2)

	# construct sentence for paper
	sentence = f"""Among these 6,125 participants, a random selection of {total} participants wore a second accelerometer, the ActiWave Cardio (CamNtech Ltd, Cambridge, UK), for at least 24 hours. Since 
					the ActiWave Cardio additionally recorded a single channel ECG waveform, correct attachment of the ECG pads was necessary to obtain valid data. Unfortunately, data from {errors} participants 
					contained measurements or device errors (e.g. axis producing too high acceleration values, no recording at all) and were removed from further analysis. A total of {n} 
					participants, {n_male} ({perc_male}%) males and {n_female} ({perc_female}%) females aged {min_age}--{max_age} (average {avg_age}, SD {std_age}), had valid data for both the 
					ActiGraph and ActiWave device. The participants had an average height of {avg_height}cm (SD={std_height}), an average weight of {avg_weight}kg (SD={std_weight}), and an average BMI of {avg_bmi}kg/m^2 (SD={std_bmi})"""

	print(sentence.replace('\t', '').replace('\n', ''))


def process_dataset_characteristics():

	# get all the subjects from the hdf5 file and remove subjects with invalid data
	subjects = [s for s in get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) if s not in get_subjects_with_invalid_data()]

	data = []
	for i, subject in enumerate(subjects):

		logging.debug(f'Processing {i}/{len(subjects)}')

		# read data
		subject_epoch_data = read_dataset_from_group(group_name = subject, dataset = 'epoch60', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE).astype('float16')

		# add to dataframe
		data.append(len(subject_epoch_data)) 

	# convert to numpy array
	data = np.array(data)

	# convert to hours
	data = data / 60

	min_hours = np.min(data)
	max_hours = np.max(data)
	avg_hours = np.mean(data)
	std_hours = np.std(data)

	logging.info(f'Min: {min_hours}, Max: {max_hours}, Average: {avg_hours}, Standard Deviation {std_hours}')


"""
	INTERNAL HELPER FUNCTIONS
"""
def extract_features_and_labels_from_files(feature_location):
	"""
	Extract features and labels from .csv files. Each .csv file contains features for 1 subject ID. CSV files can be read as a pandas dataframe

	Parameters
	----------
	feature_location : os.path
		folder location where the .csv files are stored that contain features for each participant

	Returns
	--------
	X : np.array(num of samples * num of features)
		numpy array with features describing candidate segments
	Y : np.array(num of samples * 1)
		numpy arrray with labels, that is 0 for non wear time, and 1 for wear time.
	X_labels : list
		name of each feature. See also the function get_features_for_candidate_segment for all the features
	"""

	# create empty dataframe
	df = pd.DataFrame()

	# get the .csv files with the feature
	F = [f for f in read_directory(feature_location) if f[-4:] == '.csv']

	# loop over each file, open it, and add the data to the dataframe
	for f in F:

		# open .csv file as a pandas dataframe
		df_features = pd.read_csv(f, index_col = 0)

		# add to dataframe that will contain all the features
		df = pd.concat([df, df_features], axis = 1)

		# open .pkl file (this was pickled as a pandas dataframe)
		# df = pd.concat([df, load_pickle(f)], axis = 1)

	# transpose the datframe
	df = df.T

	# get the Y labels
	Y = np.array(df.loc[:, df.columns == 'label'].values, dtype= np.int).ravel()

	# columns that we don't need for training
	filter_columns = ['start', 'stop', 'counter', 'label']

	# filter dataframe by taking all the columns except the columns defined in filter_columns
	df = df.filter( items = [c for c in df.columns if c not in filter_columns])

	# get the X features
	X = np.array(df.values, dtype= np.float)

	# get the X labels
	X_labels = df.columns.values

	# return features as X, labels as Y, and name of the features as X_labels
	return X, Y, X_labels


def get_grid_setup():
    """	
	Return ML models and hyper-parameter values for grid search

	Returns
	-------
	grid_setup: dict()
		dictionary with ML models and parameter values
	"""

    return {
        "LinearSVC": [
            {
                "hyperparameter": "C",
                "random": True,
                "min": 0.001,
                "max": 5.000,
                "length": 50,
            },
            {"hyperparameter": "dual"},
            {"hyperparameter": "fit_intercept"},
        ],
        "SVC": [
            {
                "hyperparameter": "C",
                "random": True,
                "min": 0.001,
                "max": 0.500,
                "length": 500,
            },
            {
                "hyperparameter": "gamma",
                "random": True,
                "min": 0.001,
                "max": 2.000,
                "length": 100,
            },
            {"hyperparameter": "shrinking"},
            {"hyperparameter": "kernel"},
        ],
        "LogisticRegression": [
            {
                "hyperparameter": "C",
                "random": True,
                "min": 0.001,
                "max": 10.000,
                "length": 500,
            },
            {"hyperparameter": "dual"},
            {"hyperparameter": "fit_intercept"},
        ],
        "DecisionTreeClassifier": [
            {"hyperparameter": "criterion"},
            {"hyperparameter": "splitter"},
            {"hyperparameter": "max_features"},
        ],
        "AdaBoostClassifier": [
            {
                "hyperparameter": "n_estimators",
                "random": True,
                "min": 1,
                "max": 1000,
                "length": 100,
            },
            {"hyperparameter": "algorithm"},
            {"hyperparameter": "adaboost_learning_rate", "min": 1, "max": 2},
        ],
		"MLPClassifier" : [
			{"hyperparameter" : "hidden_layer_sizes",
			"neurons" : [(10,10,10,10),
						(20,20,20,20),
						(30,30,30,30),
						(40,40,40,40),
						(50,50,50,50),
						(60,60,60,60),
						(70,70,70,70),
						(80,80,80,80),
						(90,90,90,90),
						(100,100,100,100)],
			}

		],
    }


def get_pipeline_setup():
    """
	Returns settings for the machine learning pipeline

	Returns
	-------
	pipeline_setup : dict()
		dictionary with ML classifier + pipeline options
	"""

    return {
        "LinearSVC": [
            {
                "scaler": True,
                "scaler_order": -1,
                "poly_features": True,
                "poly_features_order": -2,
                "poly_features_degree": 2,
            }
        ],
        "SVC": [
            {
                "scaler": True,
                "scaler_order": -1,
                "poly_features": True,
                "poly_features_order": -2,
                "poly_features_degree": 2,
            }
        ],
        "LogisticRegression": [
            {
                "scaler": True,
                "scaler_order": -1,
                "poly_features": True,
                "poly_features_order": -2,
                "poly_features_degree": 1,
            }
        ],
        "DecisionTreeClassifier": [
            {
                "scaler": True,
                "scaler_order": -1,
                "poly_features": True,
                "poly_features_order": -2,
                "poly_features_degree": 2,
            }
        ],
        "AdaBoostClassifier": [
            {
                "scaler": True,
                "scaler_order": -1,
                "poly_features": True,
                "poly_features_order": -2,
                "poly_features_degree": 2,
            }
        ],
		"MLPClassifier": [
            {
                "scaler": True,
                "scaler_order": -1,
                "poly_features": True,
                "poly_features_order": -2,
                "poly_features_degree": 2,
            }
        ],
    }


def get_features_for_candidate_segment(a, b, c, d, start = None, stop = None, counter = None, label = None, for_training = False):
    """
	Calculcate descriptive features for candidate segment

	Parameters
	---------
	a : pd.DataFrame()
		a = joined_segment['VMU ACTIGRAPH'].dropna().values
	b : pd.DataFrame()
		b = joined_segment['VMU ACTIWAVE'].dropna().values
	c : pd.DataFrame()
		c = joined_segment['HR ACTIWAVE'].dropna().values
	d : pd.DataFrame()
		d = joined_segment['ECG ACTIWAVE'].dropna().values
	start : datetime (optional)
		start time of the segment, only necessary when creating a labeled dataset
	stop : datetime (optional)
		stop or end time of the segment, only necessary when creating a labeled dataset
	counter : int (optional)
		numbered segment, in other words, we loop over the segments and this is basically the index of the loop (starts at zero), only necessary when creating a labeled dataset
	label : int (optional)
		0 = non wear time, and 1 = wear time, only necessary when creating a labeled dataset

	Returns
	--------
	features_for_candidate_segment : pd.Series()
		pandas series with various descriptive features
	"""

    features_for_candidate_segment = pd.Series(
        {
            "a_std": np.std(a),
            "b_std": np.std(b),
            "c_std": np.std(c),
            # 'd_std' : np.std(d),
            "ab_diff_std": np.std(b) - np.std(a),
            "a_mean": np.mean(a),
            "b_mean": np.mean(b),
            "c_mean": np.mean(c),
            # 'd_mean' : np.mean(d),
            "ab_diff_mean": np.mean(b) - np.mean(a),
            "a_min": np.min(a),
            "b_min": np.min(b),
            "c_min": np.min(c),
            # 'd_min' : np.min(d),
            "ab_diff_min": np.min(b) - np.min(a),
            "a_max": np.max(a),
            "b_max": np.max(b),
            "c_max": np.max(c),
            # 'd_max' : np.max(d),
            "ab_diff_max": np.max(b) - np.max(a),
            "a_kur": stats.kurtosis(a),
            "b_kur": stats.kurtosis(b),
            "c_kur": stats.kurtosis(c),
            # 'd_kur' : stats.kurtosis(d),
            "ab_diff_kur": stats.kurtosis(b) - stats.kurtosis(a),
            "a_signal_to_noise": signal_to_noise(a),
            "b_signal_to_noise": signal_to_noise(b),
            "c_signal_to_noise": signal_to_noise(c),
            # 'd_signal_to_noise' : signal_to_noise(d),
            "ab_diff_signal_to_noise": signal_to_noise(b) - signal_to_noise(a),
            "a_mode": stats.mode(a)[0][0],
            "b_mode": stats.mode(b)[0][0],
            "c_mode": stats.mode(c)[0][0],
            # 'd_mode' : stats.mode(d)[0][0],
            "ab_diff_mode": stats.mode(b)[0][0] - stats.mode(a)[0][0],
            "a_median": np.median(a),
            "b_median": np.median(b),
            "c_median": np.median(c),
            # 'd_median' : np.median(d),
            "ab_diff_median": np.median(b) - np.median(a),
            "a_ptp": np.ptp(a),
            "b_ptp": np.ptp(b),
            "c_ptp": np.ptp(c),
            # 'd_ptp' : np.ptp(d),
            "ab_diff_ptp": np.ptp(b) - np.ptp(a),
            "a_var": np.var(a),
            "b_var": np.var(b),
            "c_var": np.var(c),
            # 'd_var' : np.var(d),
            "ab_diff_var": np.var(b) - np.var(a),
            "ab_dtw_1": calculate_lower_bound_Keogh(b, a, 1),
        }
    )

    # if for_training is set to True, then add the following to the dictionary.
    if for_training:
        features_for_candidate_segment = features_for_candidate_segment.append(
            pd.Series(
                {"counter": counter, "label": label, "start": start, "stop": stop}
            )
        )

    return features_for_candidate_segment


if __name__ == "__main__":

	# # start timer and memory counter
	tic, process, logging = set_start()

	# 1) start mapping the actiwave data onto actigraph and find the union 
	# batch_process_mapping_actiwave_on_actigraph(use_parallel = False, skip_processed_subjects = False, limit = 10)

	# 2) batch process finding autocalibrate weights (will be saved as metadata)
	# batch_process_calculate_autocalibrate_weights(use_parallel = False, skip_processed_subjects = False, limit = 1)

	# 3a) create a labeled dataset with features for machine learning models
	# batch_process_create_features_from_labels(function = create_ml_features_from_labels, use_parallel = True)
	
	# 3a) create a labeled dataset with features for deep learning models
	# batch_process_create_features_from_labels(function = create_dl_features_from_labels, use_parallel = False, limit=1)

	# 4a) create the machine learning classifier
	# create_ml_classifier()

	# 4b) calculate confidence intervals for classifiers
	# calculate_ci_ml_classifier()

	# 4b) create deep learning classifier
	# create_dl_classifier()

	# 4c) explore_misclassification_of_training_data
	# explore_misclassification_of_training_data()

	# 5) batch process finding non-wear time in actigraph data with the help of actiwave data
	# batch_process_detect_true_non_wear_time(use_parallel = True)

	# 6a) calculate precision, recall of non wear methods to true non wear time
	# batch_process_calculate_classification_performance(function = process_calculate_classification_performance, use_parallel = False, limit = 10)
	
	# 6b save classification performance and classification data to fisk
	# process_save_classification_performance_to_file()

	# 6c) total classification performance table
	# process_classification_performance_total()

	"""
	Paper material
	"""

	# plot classification results from all participants for four non wear algorithms
	# process_plot_classification_results()
	
	# plot time distribution of data
	# process_plot_time_distribution()

	# plot three scenarios of non wear time
	# process_plot_nw_scenarios()

	# table with classification results for all, grid search and cv grid search + best parameter values
	# process_table_cv_classification()

	# get overview of distribution of non-wear times
	# process_nw_time_overview()

	# get dataset characteristics (gender, height, weight, bmi, etc)
	# process_dataset_characteristics()

	# print time and memory
	set_end(tic, process)
