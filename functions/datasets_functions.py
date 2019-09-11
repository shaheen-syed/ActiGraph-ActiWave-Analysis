# encoding:utf-8

"""
	IMPORT PACKAGES
"""
import logging
import sys
import numpy as np

"""
	IMPORT FUNCTIONS
"""
from functions.helper_functions import dictionary_values_bytes_to_string
from functions.hdf5_functions import read_dataset_from_group, read_metadata_from_group_dataset
from functions.epoch_functions import create_epoch_time_array, convert_epoch_data
from functions.autocalibrate_functions_2 import calibrate_accelerometer_data, parse_calibration_weights


def get_actigraph_acc_data(subject, hdf5_file, autocalibrate = False):
	"""
	Read actigraph acceleration data from HDF5 file, if autocalibrate is set to True, then perform autocalibration. Also create the correct
	time array

	Parameters
	---------
	subject : string
		subject ID
	hdf5_file : os.path()
		location of the HDF5 file where the data is stored
	autocalibrate: Boolean (optional)
		set to true if autocalibration need to be done

	Returns
	---------
	actigraph_acc : np.array((n_samples, axes = 3))
		actigraph acceleration data YXZ
	actigraph_meta_data : dic
		dictionary with meta data
	actigraph_time : np.array((n_samples, 1))
		time array in np.datetime
	"""
	
	try:

		# read actigraph acceleration data
		actigraph_acc = read_dataset_from_group(group_name = subject, dataset = 'actigraph_acc', hdf5_file = hdf5_file)
		# read actigraph meta-data
		actigraph_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'actigraph_acc', hdf5_file = hdf5_file)
		# convert the values of the dictionary from bytes to string
		actigraph_meta_data = dictionary_values_bytes_to_string(actigraph_meta_data)
		
		if autocalibrate:
			# parse out the weights
			actigraph_weights = parse_calibration_weights(actigraph_meta_data)
			# autocalibrate actigraph acceleration data 
			actigraph_acc = calibrate_accelerometer_data(actigraph_acc, actigraph_weights)
		# read actigraph acceleration time
		actigraph_time = np.asarray(read_dataset_from_group(group_name = subject, dataset = 'actigraph_time', hdf5_file = hdf5_file), dtype = 'datetime64[ms]')

		return actigraph_acc, actigraph_meta_data, actigraph_time

	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def get_actiwave_acc_data(subject, hdf5_file, autocalibrate = False):
	"""
	Read actiwave acceleration data from HDF5 file, if autocalibrate is set to True, then perform autocalibration. Also create the correct time array

	Parameters
	---------
	subject : string
		subject ID
	hdf5_file : os.path()
		location of the HDF5 file where the data is stored. For example ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE
	autocalibrate: Boolean (optional)
		set to true if autocalibration need to be done

	Returns
	---------
	actiwave_acc : np.array((n_samples, axes = 3))
		actiwave acceleration data YXZ
	actiwave_meta_data : dic
		dictionary with meta data
	actiwave_time : np.array((n_samples, 1))
		time array in np.datetime
	"""

	# read actiwave acceleration data
	actiwave_acc = read_dataset_from_group(group_name = subject, dataset = 'actiwave_acc', hdf5_file = hdf5_file)

	# read actigraph meta-data
	actiwave_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'actiwave_acc', hdf5_file = hdf5_file)

	# convert the values of the dictionary from bytes to string
	actiwave_meta_data = dictionary_values_bytes_to_string(actiwave_meta_data)

	if autocalibrate:
		logging.debug('Perform autocalibration of acceleration signal')
		# read actigraph meta-data
		actiwave_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = 'actiwave_acc', hdf5_file = hdf5_file)
		# parse out the weights
		actiwave_weights = parse_calibration_weights(actiwave_meta_data)
		# autocalibrate actigraph acceleration data 
		actiwave_acc = calibrate_accelerometer_data(actiwave_acc, actiwave_weights)
	
	# read actiwave acceleration time
	actiwave_time = np.asarray(read_dataset_from_group(group_name = subject, dataset = 'actiwave_time', hdf5_file = hdf5_file), dtype = 'datetime64[ns]')

	return actiwave_acc, actiwave_meta_data, actiwave_time


def get_actiwave_hr_data(subject, hdf5_file):
	"""
	Read actiwave heart rate data from HDF5 file

	Parameters
	---------
	subject : string
		subject ID
	hdf5_file : os.path()
		location of the HDF5 file where the data is stored. For example ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE

	Returns
	---------
	actiwave_hr : np.array((n_samples, 1))
		actiwave estimated heart rate data
	actiwave_hr_time : np.array((n_samples, 1))
		time array in np.datetime

	"""

	# read actiwave acceleration data
	actiwave_hr = read_dataset_from_group(group_name = subject, dataset = 'actiwave_hr', hdf5_file = hdf5_file)
	# read actiwave acceleration time
	actiwave_hr_time = np.asarray(read_dataset_from_group(group_name = subject, dataset = 'actiwave_hr_time', hdf5_file = hdf5_file), dtype = 'datetime64[ns]')

	return actiwave_hr, actiwave_hr_time

def get_actiwave_ecg_data(subject, hdf5_file):
	"""
	Read actiwave ECG data from HDF5 file

	Parameters
	---------
	subject : string
		subject ID		
	hdf5_file : os.path()
		location of the HDF5 file where the data is stored. For example ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE

	Returns
	---------
	actiwave_ecg : np.array((n_samples, 1))
		actiwave ECG data
	actiwave_ecg_time : np.array((n_samples, 1))
		time array in np.datetime
	"""

	# read actiwave acceleration data
	actiwave_ecg = read_dataset_from_group(group_name = subject, dataset = 'actiwave_ecg', hdf5_file = hdf5_file)
	# read actiwave acceleration time
	actiwave_ecg_time = np.asarray(read_dataset_from_group(group_name = subject, dataset = 'actiwave_ecg_time', hdf5_file = hdf5_file), dtype = 'datetime64[ns]')

	return actiwave_ecg, actiwave_ecg_time


def get_actigraph_epoch_data(subject, epoch_dataset, hdf5_file):
	"""
	Return 10 seconds epoch data for subject

	Parameters
	---------
	subject : string
		subject iD
	epoch_dataset : string
		name of dataset where epoch data is stored
	hdf5_file : os.path()
		location of the HDF5 file where the data is stored. For example ACTIGRAPH_HDF5_FILE

	Returns
	--------
	epoch_data : np.array((n_samples, 3 axes with XYZ counts + 1 with steps ))
		10s epoch data for the subject ID
	epoch_meta_data : dict()
		dictionary with meta data
	epoch_time_data: np.array((n_samples, 1))
		datetime array for the epoch data	
	"""

	try:

		# read epoch data
		epoch_data = read_dataset_from_group(group_name = subject, dataset = epoch_dataset, hdf5_file = hdf5_file)

		# check if subject has epoch data
		if epoch_data is None:
			logging.warning('Subject {} has no epoch data, skipping...'.format(subject))
			return None, None, None

		# read epoch meta data
		epoch_meta_data = read_metadata_from_group_dataset(group_name = subject, dataset = epoch_dataset, hdf5_file = hdf5_file)
		# convert the values of the dictionary from bytes to string
		epoch_meta_data = dictionary_values_bytes_to_string(epoch_meta_data)

		# create time array of the epoch data
		epoch_time_data = create_epoch_time_array(start_date = epoch_meta_data['Start Date'], start_date_format = epoch_meta_data['Date Format'], start_time = epoch_meta_data['Start Time'], epoch_data_length = len(epoch_data), epoch_sec = 10)

		# make epoch time data on a similar scale as raw time data (thus on ms scale and not on s)
		epoch_time_data = np.asarray(epoch_time_data, dtype='datetime64[ms]')

		return epoch_data, epoch_meta_data, epoch_time_data
	
	except Exception as e:
		
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def get_actigraph_epoch_60_data(epoch_data, epoch_time_data):
	"""
	Convert 10s epoch into 60s epoch

	Parameters
	---------
	epoch_data: np.array()
		10 sec epoch data
	epoch_time_data: np.array()
		time array for epoch data

	Returns
	-------
	epoch_60_data: np.array()
		60s epoch data
	epoch_60_time_data: np.array()
		60s epoch time array
	"""

	try:

		# convert 10 seconds epoch data to 60 seconds epoch data
		epoch_60_data = convert_epoch_data(data = epoch_data, start_epoch_sec = 10, end_epoch_sec = 60)

		# adjust the time scale, we take every 6th row, so basically every minute and we also take as many samples as we have epoch-60 vmu data, sometimes we have an additional time value due to rounding
		epoch_60_time_data = epoch_time_data[::6][:len(epoch_60_data)]

		return epoch_60_data, epoch_60_time_data

	except Exception as e:
		
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)