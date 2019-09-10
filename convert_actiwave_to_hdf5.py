# -*- coding: utf-8 -*-

"""
	IMPORTED PACKAGES
"""
import os
import glob2
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import set_start, set_end
from functions.actiwave_functions import read_edf_file, read_edf_meta_data, read_edf_channel_meta_data
from functions.hdf5_functions import save_data_to_group_hdf5, save_meta_data_to_group

"""
	GLOBAL VARIABLES
"""
# folder location of .edf actiwave files
ACTIWAVE_FOLDER = os.path.join(os.sep, 'Volumes', 'LaCie', 'Actiwave')
# HDF5 file location to store the converted .edf files to
HDF5_SAVE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIWAVE_TU7.hdf5')


def batch_process_actiwave_files(actiwave_folder = ACTIWAVE_FOLDER, use_parallel = False, num_jobs = cpu_count(), limit = None, skip_n = 0):
	"""
	batch processing of actiwave files
	- read .edf file
	- extract content and meta-data
	- save to HDF5

	Parameters
	------------
	actiwave_folder : os.path()
		folder location of the .edf actiwave files
	use_parallel : Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs : int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	limit : int (optional)
		limit the number of subjects to be processed
	skipN : int (optional)
		skip first N subjects
	"""

	logging.info('Start batch processing actiwave files')

	# get the actiwave .edf files that contain the data 
	actiwave_files = glob2.glob(os.path.join(actiwave_folder, '**', '*.edf'))[0 + skip_n:limit]

	logging.info('Number of actiwave files found: {}'.format(len(actiwave_files)))

	# if use_parallel is set to True, then use parallelization to process all files
	if use_parallel:

		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(process_actiwave_file)(f = f, i = i, total = len(actiwave_files)) for i, f in enumerate(actiwave_files))
		# execute task
		executor(tasks)

	else:

		logging.info('Processing one-by-one (parallelization off)')

		# process files one-by-one
		for i, f in enumerate(actiwave_files):

			process_actiwave_file(f = f, i = i, total = len(actiwave_files))


def process_actiwave_file(f, i = 1, total = 1,  acc_dtype = np.float32, ecg_dtype = np.float32, ms2_to_g = 0.101972, hdf5_save_location = HDF5_SAVE):

	"""
	Single processing of actiwave file
	- read .edf file
	- extract content and meta data
		- acceleration data YXZ
		- ecg data
		- estimated heart rate

	Parameters
	----------
	f : string
		file location of the .gt3x file
	i : int (optional)
		index of file to be processed, is used to display a counter of the process. Default = 1. For example, processing 12/20
	total : int (optional)
		total number of files to be processed, is used to display a counter of the process. Default = 1. For example, processing 12/20
	acc_dtype : datatype
		datatype for acceleration data. Defaults to np.float32. Meaning that each acceleration value in g is represented as 32 bit float. Can be made smaller,
		which results in less memory per value, but also less precise
	ecg_dtype : datatype
		datatype for ecg data. Defaults to np.float32. Meaning that each ecg value is represented as 32 bit float. Can be made smaller,
		which results in less memory per value, but also less precise
	ms2_to_g : float
		conversion factor to go from values measured in ms2 (meter/square second) to g (gravity)
	hdf5_save_location : os.path
		folder location where to save the extracted actiwave data to
	"""

	logging.info('Processing EDF file: {} {}/{}'.format(f, i, total))

	# read EDF data
	dic_data = read_edf_file(file = f)

	# extract edf file meta data
	edf_meta_data = read_edf_meta_data(file = f)
	
	# get subject from meta data (this is also the group name in the HDF5 file)
	subject = edf_meta_data['Patient Code']

	# check to see if the subject is also part of the file name
	if subject not in f:
		logging.error('Mismatch between subject in file name {} and within EDF meta data {}'.format(f, subject))
		return

	"""
		Process ECG data
	"""

	# read ECG data from the dictionary
	ecg_data = dic_data.get('ECG0')
	# check if ecg data available
	if ecg_data is not None:
		# reshape the array so we have a column vector
		ecg_data = ecg_data.reshape(((len(ecg_data), 1)))
		# convert the data type of the ecg
		ecg_data = ecg_data.astype(dtype = ecg_dtype)
		# read meta data for this channel
		ecg_meta_data = read_edf_channel_meta_data(file = f, channel = 0)
	else:
		logging.error('ECG data not available for file: {}'.format(f))
		return

	"""
		Process the acceleration data
	"""

	acc_x_data = dic_data.get('X')
	acc_y_data = dic_data.get('Y')
	acc_z_data = dic_data.get('Z')

	# check if X, Y, and Z have values
	if (acc_x_data is not None) and (acc_y_data is not None) and (acc_z_data is not None):

		# length of the acceleration data
		l = len(acc_x_data)
		
		# create one acceleration, original data is resized, and note that the order here is now YXZ, this is similar to the order of the raw data
		acc_data = np.hstack((	acc_y_data.reshape((l, 1)), 
								acc_x_data.reshape((l, 1)),
								acc_z_data.reshape((l, 1))))

		# convert ms^2 acceleration data into g-values
		acc_data = acc_data * ms2_to_g
		# convert acc_data to smaller float point precision
		acc_data = acc_data.astype(dtype = acc_dtype)

		# read the acceleration channel meta data (here we select channel 1, but channel 2 and 3 are also acceleration data but the contain the same values)
		acc_meta_data = read_edf_channel_meta_data(file = f, channel = 1)
	
	else:
		logging.error('Acceleration data not available for file: {}'.format(f))
		return

	"""
		Process Estimated HR data
	"""

	# read HR data
	hr_data = dic_data.get('Estimated HR')

	# check if hr data is present
	if hr_data is not None:
		# resize the array to have column vectors
		hr_data = hr_data.reshape((len(hr_data), 1))
		# read meta data for this channel
		hr_meta_data = read_edf_channel_meta_data(file = f, channel = 4)
	else:
		logging.warning('Estimated HR data not available for file: {}'.format(f))
		return
	
	"""
		Save data and meta-data to HDF5
	"""

	# save ecg data
	save_data_to_group_hdf5(group = subject, data = ecg_data, data_name = 'ecg', meta_data = ecg_meta_data, overwrite = True, create_group_if_not_exists = True, hdf5_file = hdf5_save_location)	
	# save acceleration data
	save_data_to_group_hdf5(group = subject, data = acc_data, data_name = 'acceleration', meta_data = acc_meta_data, overwrite = True, create_group_if_not_exists = True, hdf5_file = hdf5_save_location)
	# save estimated heart rate data
	save_data_to_group_hdf5(group = subject, data = hr_data, data_name = 'estimated_hr', meta_data = hr_meta_data, overwrite = True, create_group_if_not_exists = True, hdf5_file = hdf5_save_location)
	# save meta data of edf file
	save_meta_data_to_group(group_name = subject, meta_data = edf_meta_data, hdf5_file = hdf5_save_location)


if __name__ == "__main__":

	# start timer and memory counter
	tic, process, logging = set_start()

	# batch process all the actiwave files and extract data and save to HDF5 file
	batch_process_actiwave_files(use_parallel = True)

	# print time and memory
	set_end(tic, process)