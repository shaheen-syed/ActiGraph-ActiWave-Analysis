# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import os
import pandas as pd
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import set_start, set_end, calculate_vector_magnitude
from functions.hdf5_functions import get_all_subjects_hdf5, read_metadata_from_group, read_dataset_from_group, save_data_to_group_hdf5, get_datasets_from_group
from functions.datasets_functions import get_actigraph_acc_data, get_actiwave_acc_data, get_actiwave_hr_data, get_actigraph_epoch_data, get_actigraph_epoch_60_data
from functions.plot_functions import plot_non_wear_algorithms
from algorithms.non_wear_time.hecht_2009 import hecht_2009_triaxial_calculate_non_wear_time
from algorithms.non_wear_time.troiano_2007 import troiano_2007_calculate_non_wear_time
from algorithms.non_wear_time.choi_2011 import choi_2011_calculate_non_wear_time
from algorithms.non_wear_time.hees_2013 import hees_2013_calculate_non_wear_time

"""
	GLOBAL VARIABLES
"""
ACTIGRAPH_HDF5_FILE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIGRAPH_TU7.hdf5')
ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIWAVE_ACTIGRAPH_MAPPING.hdf5')


def batch_process_non_wear_algorithm(algorithm, limit = None, skip_n = 0, use_parallel = False, num_jobs = cpu_count(), save_hdf5 = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE):
	"""
	Batch process finding non-wear time based on the following algorithms:
		- hecht_2009_triaxial_calculate_non_wear_time
		- troiano_2007_calculate_non_wear_time
		- choi_2011_calculate_non_wear_time
		- hees_2013_calculate_non_wear_time

	Parameters
	-----------
	algorithm : python function name
		name of the function that processes the non-wear time method
	limit : int (optional)
		limit the number of subjects to be processed
	skip_n : int (optional)
		skip first N subjects
	use_parallel : Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs : int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	save_hdf5 : os.path
		location of HDF5 file to save data to
	"""

	# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
	subjects = get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)[0 + skip_n:limit]

	logging.info('Start batch processing estimating non-wear time based on {}'.format(algorithm.__name__))

	# loop over the subjects
	if use_parallel:

		# verbose
		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(algorithm)(subject = subject, idx = i, total = len(subjects), save_hdf5 = save_hdf5) for i, subject in enumerate(subjects))
		# execute task
		executor(tasks)

	else:

		# verbose
		logging.info('Processing one-by-one (parallelization off)')

		# loop over the subjects
		for i, subject in enumerate(subjects):

			algorithm(subject = subject, idx = i, total = len(subjects), save_hdf5 = save_hdf5)


def process_hecht_2009_triaxial(subject, save_hdf5, idx = 1, total = 1, epoch_dataset = 'epoch10'):
	"""
	Calculate the non-wear time from a data array that contains the vector magnitude (VMU) according to Hecht 2009 algorithm

	Paper:
	COPD. 2009 Apr;6(2):121-9. doi: 10.1080/15412550902755044.
	Methodology for using long-term accelerometry monitoring to describe daily activity patterns in COPD.
	Hecht A1, Ma S, Porszasz J, Casaburi R; COPD Clinical Research Network.

	Parameters
	---------
	subject : string
		subject ID
	save_hdf5 : os.path
		location of HDF5 file to save non wear data to
	idx : int (optional)
		index of counter, only useful when processing large batches and you want to monitor the status
	total: int (optional)
		total number of subjects to process, only useful when processing large batches and you want to monitor the status
	epoch_dataset : string (optional)
		name of dataset within an HDF5 group that contains the 10sec epoch data
	"""

	logging.info('{style} Processing subject: {} {}/{} {style}'.format(subject, idx, total, style = '='*10))
	
	"""
		ACTIGRAPH DATA
	"""

	# read actigraph acceleration time
	_, _, actigraph_time = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	
	# get start and stop time
	start_time, stop_time = actigraph_time[0], actigraph_time[-1]

	"""
		EPOCH DATA
	"""

	# check if epoch dataset is part of HDF5 group
	if epoch_dataset in get_datasets_from_group(group_name = subject, hdf5_file = ACTIGRAPH_HDF5_FILE):

		# get actigraph 10s epoch data
		epoch_data, _ , epoch_time_data = get_actigraph_epoch_data(subject, epoch_dataset = epoch_dataset, hdf5_file = ACTIGRAPH_HDF5_FILE)

		# convert to 60s epoch data	
		epoch_60_data, epoch_60_time_data = get_actigraph_epoch_60_data(epoch_data, epoch_time_data)

		# calculate epoch 60 VMU
		epoch_60_vmu_data = calculate_vector_magnitude(epoch_60_data[:,:3], minus_one = False, round_negative_to_zero = False)


		"""
			GET NON WEAR VECTOR
		"""

		# create dataframe of actigraph acceleration 
		df_epoch_60_vmu = pd.DataFrame(epoch_60_vmu_data, index = epoch_60_time_data, columns = ['VMU']).loc[start_time:stop_time]

		# retrieve non-wear vector
		epoch_60_vmu_non_wear_vector = hecht_2009_triaxial_calculate_non_wear_time(data = df_epoch_60_vmu.values)

		# get the croped time array as int64 (cropped because we selected the previous dataframe to be between start and stop slice)
		epoch_60_time_data_cropped = np.array(df_epoch_60_vmu.index).astype('int64')
		# reshape
		epoch_60_time_data_cropped = epoch_60_time_data_cropped.reshape(len(epoch_60_time_data_cropped), 1)

		# add two arrays
		combined_data = np.hstack((epoch_60_time_data_cropped, epoch_60_vmu_non_wear_vector))
		
		"""
			SAVE TO HDF5 FILE
		"""
		save_data_to_group_hdf5(group = subject, data = combined_data, data_name = 'hecht_2009_3_axes_non_wear_data', overwrite = True, create_group_if_not_exists = False, hdf5_file = save_hdf5)

	else:
		logging.warning('Subject {} has no corresponding epoch data, skipping...'.format(subject))


def process_troiano_2007(subject, save_hdf5, idx = 1, total = 1, epoch_dataset = 'epoch10'):
	"""
	Calculate non wear time by using Troiano 2007 algorithm

	Troiano 2007 non-wear algorithm
		detects non wear time from 60s epoch counts
		Nonwear was defined by an interval of at least 60 consecutive minutes of zero activity intensity counts, with allowance for 1–2 min of counts between 0 and 100
	Paper:
		Physical Activity in the United States Measured by Accelerometer
	DOI:
		10.1249/mss.0b013e31815a51b3

	Parameters
	---------
	subject : string
		subject ID
	save_hdf5 : os.path
		location of HDF5 file to save non wear data to
	idx : int (optional)
		index of counter, only useful when processing large batches and you want to monitor the status
	total: int (optional)
		total number of subjects to process, only useful when processing large batches and you want to monitor the status
	epoch_dataset : string (optional)
		name of dataset within an HDF5 group that contains the 10sec epoch data
	"""

	logging.info('{style} Processing subject: {} {}/{} {style}'.format(subject, idx, total, style = '='*10))

	"""
		ACTIGRAPH DATA
	"""

	# read actigraph acceleration time
	_, _, actigraph_time = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

	# get start and stop time
	start_time, stop_time = actigraph_time[0], actigraph_time[-1]


	"""
		EPOCH DATA
	"""
	if epoch_dataset in get_datasets_from_group(group_name = subject, hdf5_file = ACTIGRAPH_HDF5_FILE):

		# get actigraph 10s epoch data
		epoch_data, _ , epoch_time_data = get_actigraph_epoch_data(subject, epoch_dataset = epoch_dataset, hdf5_file = ACTIGRAPH_HDF5_FILE)

		# convert to 60s epoch data	
		epoch_60_data, epoch_60_time_data = get_actigraph_epoch_60_data(epoch_data, epoch_time_data)

		# obtain counts values
		epoch_60_count_data = epoch_60_data[:,:3]

		"""
			GET NON WEAR VECTOR
		"""

		# create dataframe of actigraph acceleration 
		df_epoch_60_count = pd.DataFrame(epoch_60_count_data, index = epoch_60_time_data, columns = ['X - COUNT', 'Y - COUNT', 'Z - COUNT']).loc[start_time:stop_time]

		# retrieve non-wear vector
		epoch_60_count_non_wear_vector = troiano_2007_calculate_non_wear_time(data = df_epoch_60_count.values, time = df_epoch_60_count.index.values)

		# get the croped time array as int64 (cropped because we selected the previous dataframe to be between start and stop slice)
		epoch_60_time_data_cropped = np.array(df_epoch_60_count.index).astype('int64')
		# reshape
		epoch_60_time_data_cropped = epoch_60_time_data_cropped.reshape(len(epoch_60_time_data_cropped), 1)

		# add two arrays
		combined_data = np.hstack((epoch_60_time_data_cropped, epoch_60_count_non_wear_vector))

		
		"""
			SAVE TO HDF5 FILE
		"""
	
		save_data_to_group_hdf5(group = subject, data = combined_data, data_name = 'troiano_2007_non_wear_data', overwrite = True, create_group_if_not_exists = False, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

	else:
		logging.warning('Subject {} has no corresponding epoch data, skipping...'.format(subject))


def process_choi_2011(subject, save_hdf5, idx = 1, total = 1, epoch_dataset = 'epoch10'):
	"""
	Estimate non-wear time based on Choi 2011 paper:

	Med Sci Sports Exerc. 2011 Feb;43(2):357-64. doi: 10.1249/MSS.0b013e3181ed61a3.
	Validation of accelerometer wear and nonwear time classification algorithm.
	Choi L1, Liu Z, Matthews CE, Buchowski MS.

	Parameters
	---------
	subject : string
		subject ID
	save_hdf5 : os.path
		location of HDF5 file to save non wear data to
	idx : int (optional)
		index of counter, only useful when processing large batches and you want to monitor the status
	total: int (optional)
		total number of subjects to process, only useful when processing large batches and you want to monitor the status
	epoch_dataset : string (optional)
		name of dataset within an HDF5 group that contains the 10sec epoch data
	"""

	logging.info('{style} Processing subject: {} {}/{} {style}'.format(subject, idx, total, style = '='*10))

	"""
		ACTIGRAPH DATA
	"""

	# read actigraph acceleration time
	_, _, actigraph_time = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

	# get start and stop time
	start_time, stop_time = actigraph_time[0], actigraph_time[-1]


	"""
		EPOCH DATA
	"""
	if epoch_dataset in get_datasets_from_group(group_name = subject, hdf5_file = ACTIGRAPH_HDF5_FILE):

		# get actigraph 10s epoch data
		epoch_data, _ , epoch_time_data = get_actigraph_epoch_data(subject, epoch_dataset = epoch_dataset, hdf5_file = ACTIGRAPH_HDF5_FILE)

		# convert to 60s epoch data	
		epoch_60_data, epoch_60_time_data = get_actigraph_epoch_60_data(epoch_data, epoch_time_data)

		# obtain counts values
		epoch_60_count_data = epoch_60_data[:,:3]

		"""
			GET NON WEAR VECTOR
		"""

		# create dataframe of actigraph acceleration 
		df_epoch_60_count = pd.DataFrame(epoch_60_count_data, index = epoch_60_time_data, columns = ['X - COUNT', 'Y - COUNT', 'Z - COUNT']).loc[start_time:stop_time]

		# retrieve non-wear vector
		epoch_60_count_non_wear_vector = choi_2011_calculate_non_wear_time(data = df_epoch_60_count.values, time = df_epoch_60_count.index.values)

		# get the croped time array as int64 (cropped because we selected the previous dataframe to be between start and stop slice)
		epoch_60_time_data_cropped = np.array(df_epoch_60_count.index).astype('int64')
		
		# reshape
		epoch_60_time_data_cropped = epoch_60_time_data_cropped.reshape(len(epoch_60_time_data_cropped), 1)

		# add two arrays
		combined_data = np.hstack((epoch_60_time_data_cropped, epoch_60_count_non_wear_vector))

		
		"""
			SAVE TO HDF5 FILE
		"""
		
		save_data_to_group_hdf5(group = subject, data = combined_data, data_name = 'choi_2011_non_wear_data', overwrite = True, create_group_if_not_exists = False, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

	else:
		logging.warning('Subject {} has no corresponding epoch data, skipping...'.format(subject))


def process_hees_2013(subject, save_hdf5,  idx = 1, total = 1):
	"""
	Estimation of non-wear time periods based on Hees 2013 paper

	Estimation of Daily Energy Expenditure in Pregnant and Non-Pregnant Women Using a Wrist-Worn Tri-Axial Accelerometer
	Vincent T. van Hees, Frida Renström , Antony Wright, Anna Gradmark, Michael Catt, Kong Y. Chen, Marie Löf, Les Bluck, Jeremy Pomeroy, Nicholas J. Wareham, Ulf Ekelund, Søren Brage, Paul W. Franks
	Published: July 29, 2011https://doi.org/10.1371/journal.pone.0022922

	Accelerometer non-wear time was estimated on the basis of the standard deviation and the value range of each accelerometer axis, calculated for consecutive blocks of 30 minutes. 
	A block was classified as non-wear time if the standard deviation was less than 3.0 mg (1 mg = 0.00981 m·s−2) for at least two out of the three axes or if the value range, for 
	at least two out of three axes, was less than 50 mg.

	Parameters
	---------
	subject : string
		subject ID
	save_hdf5 : os.path
		location of HDF5 file to save non wear data to
	idx : int (optional)
		index of counter, only useful when processing large batches and you want to monitor the status
	total: int (optional)
		total number of subjects to process, only useful when processing large batches and you want to monitor the status
	"""

	logging.info('{style} Processing subject: {} {}/{} {style}'.format(subject, idx, total, style = '='*10))
	
	# read actigraph acceleration data
	actigraph_acc, *_ = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

	# calculate non non wear time based on Hees 2013 algorithm
	non_wear_vector = hees_2013_calculate_non_wear_time(actigraph_acc)
	
	# save non-wear vector to HDF5
	save_data_to_group_hdf5(group = subject, data = non_wear_vector, data_name = 'hees_2013_non_wear_data', overwrite = True, create_group_if_not_exists = False, hdf5_file = save_hdf5)


def batch_process_plot_non_wear_algorithms(limit = None, skip_n = 0, plot_folder = os.path.join('plots', 'non-wear-time', 'algorithms')):
	"""
	Batch process finding non-wear time based on Hecht 2009 algorithm

	Parameters
	-----------
	limit : int (optional)
		limit the number of subjects to be processed
	skip_n : int (optional)
		skip first N subjects
	"""

	# get all the subjects from the hdf5 file (subjects are individuals who participated in the Tromso Study #7
	subjects = get_all_subjects_hdf5(hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)[0 + skip_n:limit]
	
	# loop over the subjects
	for i, subject in enumerate(subjects):

		# verbose
		logging.info('Processing subject: {} {}/{}'.format(subject, i, len(subjects)))

		# call plot function
		process_plot_non_wear_algorithms(subject, plot_folder)


def process_plot_non_wear_algorithms(subject, plot_folder, epoch_dataset = 'epoch10'):
	"""
	Plot acceleration data from actigraph and actiwave including the 4 non wear methods and the true non wear time

	- plot actigraph XYZ
	- plot actiwave YXZ
	- plot actiwave heart rate
	- plot hecht, choi, troiano, v. Hees
	- plot true non wear time

	Parameters
	---------
	subject : string
		subject ID
	plot_folder : os.path
		folder location to save plots to
	epoch_dataset : string (optional)
		name of hdf5 dataset that contains 10s epoch data
	"""


	"""
		GET ACTIGRAPH DATA
	"""
	actigraph_acc, _ , actigraph_time = get_actigraph_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	# get start and stop time
	start_time, stop_time = actigraph_time[0], actigraph_time[-1]

	"""
		GET ACTIWAVE DATA
	"""
	actiwave_acc, _ , actiwave_time = get_actiwave_acc_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
	actiwave_hr, actiwave_hr_time = get_actiwave_hr_data(subject, hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)

	"""
		EPOCH DATA
	"""

	if epoch_dataset in get_datasets_from_group(group_name = subject, hdf5_file = ACTIGRAPH_HDF5_FILE):

		# get 10s epoch data from HDF5 file
		epoch_data, _ , epoch_time_data = get_actigraph_epoch_data(subject, epoch_dataset = epoch_dataset, hdf5_file = ACTIGRAPH_HDF5_FILE)

		# convert to 60s epoch data	
		epoch_60_data, epoch_60_time_data = get_actigraph_epoch_60_data(epoch_data, epoch_time_data)

		# calculate epoch 60 VMU
		epoch_60_vmu_data = calculate_vector_magnitude(epoch_60_data[:,:3], minus_one = False, round_negative_to_zero = False)


		"""
			GET NON WEAR TIME
		"""

		# true non wear time
		true_non_wear_time = read_dataset_from_group(group_name = subject, dataset = 'actigraph_true_non_wear', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) 
		# hecht 3-axes non wear time
		hecht_3_non_wear_time = read_dataset_from_group(group_name = subject, dataset = 'hecht_2009_3_axes_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) 
		# troiano non wear time
		troiano_non_wear_time = read_dataset_from_group(group_name = subject, dataset = 'troiano_2007_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)
		# choi non wear time
		choi_non_wear_time = read_dataset_from_group(group_name = subject, dataset = 'choi_2011_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE) 
		# hees non wear time
		hees_non_wear_time = read_dataset_from_group(group_name = subject, dataset = 'hees_2013_non_wear_data', hdf5_file = ACTIWAVE_ACTIGRAPH_MAPPING_HDF5_FILE)


		"""
			CREATING THE DATAFRAMES
		"""

		# convert actigraph data to pandas dataframe
		df_actigraph_acc = pd.DataFrame(actigraph_acc, index = actigraph_time, columns = ['ACTIGRAPH Y', 'ACTIGRAPH X', 'ACTIGRAPH Z'])
		# convert actiwave data to pandas dataframe
		df_actiwave_acc = pd.DataFrame(actiwave_acc, index = actiwave_time, columns = ['ACTIWAVE Y', 'ACTIWAVE X', 'ACTIWAVE Z'])
		# convert actiwave hr to pandas dataframe
		df_actiwave_hr = pd.DataFrame(actiwave_hr, index = actiwave_hr_time, columns = ['ESTIMATED HR'])
		# convert 60s epoch vmu to dataframe
		df_epoch_60_vmu = pd.DataFrame(epoch_60_vmu_data, index = epoch_60_time_data, columns = ['EPOCH 60s VMU'])
		# slice based on start and stop time
		df_epoch_60_vmu = df_epoch_60_vmu.loc[start_time:stop_time]
		# create dataframe of true non wear time
		df_true_non_wear_time = pd.DataFrame(true_non_wear_time, index = actigraph_time, columns = ['TRUE NON WEAR TIME'])
		# create dataframe of hecht 3-axes non wear time
		df_hecht_3_non_wear_time = pd.DataFrame(hecht_3_non_wear_time[:,1], index = np.asarray(hecht_3_non_wear_time[:,0], dtype ='datetime64[ns]'), columns = ['HECHT-3 NON WEAR TIME'])
		# create dataframe of troiano non wear time
		df_troiano_non_wear_time = pd.DataFrame(troiano_non_wear_time[:,1], index = np.asarray(troiano_non_wear_time[:,0], dtype = 'datetime64[ns]'), columns = ['TROIANO NON WEAR TIME'])
		# create dataframe of choi non wear time
		df_choi_non_wear_time = pd.DataFrame(choi_non_wear_time[:,1], index = np.asarray(choi_non_wear_time[:,0], dtype = 'datetime64[ns]'), columns = ['CHOI NON WEAR TIME'])
		# create dataframe of hees non wear time
		df_hees_non_wear_time = pd.DataFrame(hees_non_wear_time, index = actigraph_time, columns = ['HEES NON WEAR TIME'])
		
		# merge all dataframes
		df_joined = df_actigraph_acc.join(df_true_non_wear_time, how='outer').join(df_actiwave_acc, how='outer').join(df_hecht_3_non_wear_time, how='outer') \
					.join(df_troiano_non_wear_time, how='outer').join(df_choi_non_wear_time, how='outer').join(df_hees_non_wear_time, how='outer').join(df_epoch_60_vmu, how='outer').join(df_actiwave_hr, how='outer')

		# call plot function
		plot_non_wear_algorithms(data = df_joined, subject = subject, plot_folder = plot_folder)


if __name__ == "__main__":

	# start timer and memory counter
	tic, process, logging = set_start()

	# 1) batch process Hecht 2009 non-wear method
	batch_process_non_wear_algorithm(algorithm = process_hecht_2009_triaxial)

	# 2) batch process Troiano 2007 non-wear method
	batch_process_non_wear_algorithm(algorithm = process_troiano_2007)

	# 3) batch process Choi 2011 non-wear method
	batch_process_non_wear_algorithm(algorithm = process_choi_2011)

	# 4) batch process Hees 2013 non-wear method
	batch_process_non_wear_algorithm(algorithm = process_hees_2013)
	
	# 5) batch process the plotting of the non-wear algorithms including true non-wear time
	batch_process_plot_non_wear_algorithms(skip_n = 103)

	# print time and memory
	set_end(tic, process)
