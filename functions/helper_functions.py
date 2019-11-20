# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import logging
import os
import glob2
import sys
import csv
import time
import psutil
import random
import shutil
import pickle
import numpy as np
import warnings
from datetime import datetime


def set_logger(folder_name = 'logs'):
	"""
	Set up the logging to console layout

	Parameters
	----------
	folder_name : string, optional
		name of the folder where the logs can be saved to

	Returns
	--------
	logger: logging
		logger to console and file
	"""

	# create the logging folder if not exists
	create_directory(folder_name)

	# omit logging messages from matplotlib
	logging.getLogger('matplotlib').setLevel(logging.ERROR)

	# define the name of the log file
	log_file_name = os.path.join(folder_name, '{:%Y%m%d%H%M%S}.log'.format(datetime.now()))

	# create a new logger but use root
	logger = logging.getLogger('')

	# clear existing handlers to avoid duplicated output
	logger.handlers.clear()

	# set logging level, DEBUG means everything, also INFO, WARNING, EXCEPTION, ERROR etc
	logger.setLevel(logging.DEBUG)

	# define the format of the message
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

	# write log to filehandler
	file_handler = logging.FileHandler(log_file_name)
	file_handler.setFormatter(formatter)

	# write to console
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)

	# add stream handler and file handler to logger
	logger.addHandler(stream_handler)
	logger.addHandler(file_handler)

	# tensorflow doubles the messages, so we turn them off here
	logger.propagate = False
	
	return logger


def set_start():
	"""
	Make sure the logger outputs to the console in a certain format
	Define the start time and get the process so we know how much memory we are using

	Returns
	----------
	tic : timestamp
		time the program starts
	process : object
		process id
	logger : logging object
		logger that outputs to console and save log file to disk
	"""

	# create logging to console
	logger = set_logger()

	# define start time
	tic = time.time()

	# define process ID
	process = psutil.Process(os.getpid())

	return tic, process, logger


def set_end(tic, process):
	"""
	Verbose function to display the elapsed time and used memory

	Parameters
	----------
	tic : timestamp
		time the program has started
	"""

	# print time elapsed
	logging.info('-- executed in {} seconds'.format(time.time()-tic))
	logging.info('-- used {} MB of memory'.format(process.memory_info().rss / 1024 / 1024))


def create_directory(name):
	"""
	Create directory if not exists

	Parameters
	----------
	name : string
		name of the folder to be created
	"""

	try:
		if not os.path.exists(name):
			os.makedirs(name)
			logging.info('Created directory: {}'.format(name))
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_directory(directory):
	"""
	Read file names from directory recursively

	Parameters
	----------
	directory : string
		directory/folder name where to read the file names from

	Returns
	---------
	files : list of strings
		list of file names
	"""
	
	try:
		return glob2.glob(os.path.join( directory, '**' , '*.*'))
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def delete_directory(directory):
	"""
	Delete a directory and all the underlying content

	Parameters
	----------
	directory : string
		path of directory
	"""

	try:
		shutil.rmtree(directory)
		logging.info('Deleted directory: {}'.format(directory))
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def delete_file(file):
	"""
	Delete a file

	Parameters
	----------
	file : string
		path of file
	"""

	try:
		os.remove(file)
		logging.info('Deleted file: {}'.format(file))
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def save_pickle(obj, file_name, folder):	
	"""
	Save python object with pickle

	Parameters
	----------
	obj : object
		object that need to be pickled
	name : string
		name of the file
	folder : string
		location of folder to store pickle file in
	"""

	# create folder if not exists
	create_directory(folder)

	# check if .pkl is used as an extension, this is not required
	if file_name[-4:] == '.pkl':
		file_name = file_name[:-4]

	# save as pickle
	with open(os.path.join(folder, file_name + '.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name, folder = None):
	"""
	Load python object with pickle

	Parameters
	---------
	file_name : string
		file name of the pickle file to load/open
	folder : string (optional)
		name of folder if not already part of the file name

	Returns
	-------
	pickle file: pickle object
	"""

	# check if .pkl is used as an extension, this is not required
	if file_name[-4:] == '.pkl':
		file_name = file_name[:-4]

	# check if folder has been sent
	if folder is not None:
		file_name = os.path.join(folder, file_name)

	# open file with pickle and return
	with open(os.path.join(file_name + '.pkl'), 'rb') as f:
		return pickle.load(f)


def save_csv(data, name, folder):
	"""
	Save list of list as CSV (comma separated values)

	Parameters
	----------
	data : list of list
		A list of lists that contain data to be stored into a CSV file format
	name : string
		The name of the file you want to give it
	folder: string
		The folder location
	"""
	
	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# create the path name (allows for .csv and no .csv extension to be handled correctly)
		suffix = '.csv'
		if name[-4:] != suffix:
			name += suffix

		# create the file name
		path = os.path.join(folder, name)

		# save data to folder with name
		with open(path, "w") as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(data)

	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_csv(filename, folder = None):
	"""
	Read CSV file and return as a list

	Parameters
	---------
	filename : string
		name of the csv file
	folder : string (optional)
		name of the folder where the csv file can be read
	"""

	if folder is not None:
		filename = os.path.join(folder, filename)
	
	try:
		# increate CSV max size
		csv.field_size_limit(sys.maxsize)
		
		# open the filename
		with open(filename, 'rt') as f:
			# create the reader
			reader = csv.reader(f)
			# return csv as list
			return list(reader)
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))


def get_current_timestamp():
	"""
	Return the current timestamp in string format

	Returns
	--------
	current_timestamp : string
		string of time with year-month-day-hour-minute-seconds
	"""

	return '{:%Y%m%d%H%M%S}'.format(datetime.now())


def get_date_format_parser(date_format):
	"""
	Based on the date format, return the corresponding parser

	Parameters
	----------
	date_format : string
		the format of the date, for instance M/d/yyyy or dd.MM.yyyy

	Returns
	-------
	date_format_parser: string
		string to extract correct date elements from a string
	"""

	if date_format == 'dd.MM.yyyy':

		return '%d.%m.%Y'

	elif date_format == 'M/d/yyyy':

		return '%m/%d/%Y'
	
	else:
		logging.error('Unknown date format {}, no parser available'.format(date_format))
		exit(1)


def get_random_number_between(x, y):
	"""
	Return a random number between two values

	Parameters
	---------
	x : int
		Lower bound of the random number
	y : int
		Upper bound of the random number

	Returns
	--------
	value : int
		random number between x and y
	"""

	return random.randint(x,y)


def scale_values_between(data, a = -1., b = 1.):
	"""
	Scale numpy array
	In general, to scale your variable x into a range [a,b] you can use:
	xnormalized=(b−a)x−min(x)max(x)−min(x)+a

	Parameters
	----------
	a : int
		lower bound of the scaling
	b : int
		upper bound of the scaling
	data : np.array
		numpy array with the data

	Returns
	--------
	scaled numpy array between [a,b]
	"""

	return (b-a) * (data - np.min(data)) / (np.ptp(data)) + a


def calculate_vector_magnitude(data, minus_one = False, round_negative_to_zero = False, dtype = np.float32):
	"""
	Calculate vector magnitude of acceleration data
	the vector magnitude of acceleration is calculated as the Euclidian Norm

	sqrt(y^2 + x^2 + z^2)

	if minus_one is set to True then it it is the Euclidian Norm Minus One

	sqrt(y^2 + x^2 + z^2) - 1

	Parameters
	----------
	data : numpy array (acceleration values, axes)
		numpy array with acceleration data
	minus_one : Boolean (optional)
		If set to True, the calculate the vector magnitude minus one, also known as the ENMO (Euclidian Norm Minus One)
	round_negative_to_zero : Boolean (optional)
		If set to True, round negative values to zero
	dtype = mumpy data type (optional)
		set the data type of the return array. Standard float 16, but can be set to better precision
	
	Returns
	-------
	vector_magnitude : numpy array (acceleration values, 1)(np.float)
		numpy array with vector magnitude of the acceleration
	"""

	# change dtype of array to float32 (also to hold scaled data correctly). The original unscaled data is stored as int16, but when we want to calculate the vector we exceed the values that can be stored in 16 bit
	data = data.astype(dtype = np.float32)

	try:

		# calculate the vector magnitude on the whole array
		vector_magnitude = np.sqrt(np.sum(np.square(data), axis=1)).astype(dtype=dtype)

		# check if minus_one is set to True, if so, we need to calculate the ENMO
		if minus_one:
			vector_magnitude -= 1

		# if set to True, round negative values to zero
		if round_negative_to_zero:
			vector_magnitude = vector_magnitude.clip(min=0)

		# reshape the array into number of acceleration values, 1 column
		return vector_magnitude.reshape(data.shape[0], 1)
		

	except Exception as e:
		
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def calculate_moving_average(data, n = 2):
	"""
	Calculate moving average of array

	Parameters
	----------
	data : np.array(n_samples, 1)
		numpy array with some numeric value
	n : int (optional)
		window size

	Returns
	--------
	ret : float
		moving average value
	"""

	ret = np.cumsum(data, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def save_dic_to_csv(dic, file_name, folder):
	"""
	Save a dictionary as CSV (comma separated values)

	Parameters
	----------
	dic : dic
		dictionary with key value pairs
	name : string
		The name of the file you want to give it
	folder: string
		The folder location
	"""
	
	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# check if .csv is used as an extension, this is not required
		if file_name[-4:] == '.csv':
			file_name = file_name[:-4]

		# create the file name
		file_name = os.path.join(folder, file_name + '.csv')

		# save data to folder with name
		with open(file_name, "w") as f:

			writer = csv.writer(f, lineterminator='\n')
			
			for k, v in dic.items():
				writer.writerow([k, v])

	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def dictionary_values_bytes_to_string(dictionary):
	"""
	Convert the values of a dictionary to strings from bytes

	Parameters
	---------
	dictionary: dic
		dictionary with values in byte 

	returns
	---------
	dictionary : dic
		the same dictionary as the input parameter but not the values have been changed to strings
	"""

	for key, value in dictionary.items():

		if isinstance(value, bytes):

			dictionary[key] = value.decode('utf-8')

	return dictionary

def get_subjects_with_invalid_data():
	"""
	Return a list of subjects that need to be excluded from further analysis. These subjects have measurements error in their data; could be caused by faulty sensor
	or improper placement on the subject

	Returns
	---------
	invalid_subjects : list
		list of subjects that need to be excluded
	"""

	# subjects that should be excluded from further analysis due to invalid data
	invalid_subjects = ['90013013', # actiwave data invalid
						'90022619',# actiwave data invalid
						'90097429', # actiwave z axis too high
						'90107724', # actiwave data flat
						'90265628', # actiwave data flat
						'90277429', # actiwave data measurement errors in beginning
						'90352827', # actiwave y and z invalid
						'90358126', # actiwave data flat
						'90359935', # actiwave data measurement errors
						'90551323', # actiwave flat
						'90631928', # actiwave measurements error
						'90821020', # actiwave measurements error
						'90884635', # actiwave measurements error
						'90936734', # actiwave flat
						'90952429', # actiwave measurements error
						'92108626', # actiwave flat
						'92280425', # actiwave measurements error
						'92327831', # actiwave flat
						'92355428', # actiwave flat
						'92504323', # actiwave measurements error
						'93208931', # actiwave measurements error
						'90165829'	# no epoch data
						]

	return invalid_subjects


def get_subject_counters_for_correction(subject):

	"""
	get a list of subjects counters to correct. The counter represents a candidate segment for non wear time. If subject is listed here, it means that the classifier has inferred an incorrect label for a specific segment.
	These segments are indicated by their counter, with the first segment being 0 etc. If subject + counter is part of the dictionary, the labels will be flipped typically. So what is inferred as non wear time, need to be wear time and
	vice versa.

	IMPORTANT : counters are basically non wear segments, but changing the parameter values when finding segments, i.e. calling function find_candidate_non_wear_segments_from_raw, can provide different segments and thus different counters.
	counters here are based on std_threshold = 0.004, min_segment_length = 1, sliding_window = 1

	Parameters
	---------
	subject : string
		subject ID
	
	Returns
	--------
	subject_counter_to_correct : list or None
		if subject is part of the dictionary, it returns a list with counters to correct for. If the subject is not part of the dictionary, it returns None.
	"""

	subject_counter_to_correct = {	'90086831' : [0,1,2,3,4,5,6,7],
									'91864028' : [38],
									'90974029' : [4],
									'90173929' : [0],
									'90187631' : [0],
									'90475732' : [0],
									'90656834' : [2],
									'90897538' : [25],
									'92460728' : [0],
									'92987439' : [1]
									}

	# return counters to correct, if not part of dictionary, then it returs None; meaning no correction to apply
	return subject_counter_to_correct.get(subject)

def convert_short_code_to_long(short_code):

	labels = {	'T' : 'Threshold (VMU)', 
				'I' : 'Window size (mins)', 
				'M' : 'Allowed #spikes in window',
				'AT' : 'Minimum spike threshold (counts)', 
				'MPL' : 'Minimum interval (mins)', 
				'ST' : 'Allowed #spikes',
				'SS' : 'Maximum spike threshold (counts)',
				'MWL' : 'Window size (mins)',
				'WST' : 'Allowed #spikes in window',
				'MW' : 'Minimum interval (mins)', 
				'WO' : 'Window overlap',
				'StdT' : 'Std. threshold (mg)',
				'StdA' : 'Std. min. #axes',
				'VT' : 'Value threshold (mg)',
				'VA' : 'Value min. #axes',
				'VM' : 'Use VMU'
			}
	
	return labels[short_code]
				