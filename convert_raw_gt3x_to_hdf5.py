# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import glob2
import re
import time
import os
import logging
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import set_start, set_end, delete_file, delete_directory
from functions.gt3x_functions import unzip_gt3x_file, extract_info, extract_log
from functions.hdf5_functions import get_all_subjects_hdf5, read_metadata_from_group, read_dataset_from_group, read_metadata_from_group_dataset, save_multi_data_to_group_hdf5, save_meta_data_to_group_dataset, save_data_to_group_hdf5

"""
CHANGE LOG

19-02-2019 changed location of functions (now in folder functions) Shaheen Syed 
24-06-2019 added batch processing and the process_gt3x_file function is now part of this file, and not part of gt3x_functions
"""

"""
	GLOBAL VARIABLES
"""
# folder location of the .gt3x files
GT3X_FOLDER = os.path.join(os.sep, 'Volumes', 'LaCie_server', 'Actigraph_raw')
# folder location to store data in HDF5 format to
HDF5_SAVE = os.path.join(os.sep, 'Volumes', 'LaCie_server', 'ACTIGRAPH_TU7.hdf5')


def batch_process_gt3x_files(gt3x_folder = GT3X_FOLDER, use_parallel = False, num_jobs = cpu_count(), limit = None, skip_n = 0, ignore_already_processed_subjects = True, remove_unwanted_files = True):
	"""
	Batch processing to convert actigraph .gt3x (raw acceleration) files into YXZ acceleration in g, and also the corresponding time array so we know the start and stop times of the signal

	Parameters
	------------
	gt3x_folder : os.path()
		folder location of the (raw) gt3x files.
	use_parallel = Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs = int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	limit : int (optional)
		limit the number of subjects to be processed
	skipN : int (optional)
		skip first N subjects
	skip_processed_subjects : Boolean (optional)
		skip subjects that are already part of the target hdf5 file
	ignore_already_processed_subjects : Boolean (optional)
		set to True if already processed subjects need to be removed from the files
	remove_unwanted_files : Boolean (optional)
		set to True if you want to delete any files prior to processing the full folder (in case of interuptions there might be half unzipped log files)
	"""


	# get all the .gt3x files from the folder location. We do this here because there might also be other types of files in the folder
	# we can also skip_n the first n files, or it is possible to limit the number of files to be processed, such for testing or if we only need, for example, 100 files
	gt3x_files = glob2.glob(os.path.join(gt3x_folder, '**', '*.gt3x'))[0 + skip_n:limit]

	# if set to True, remove the files that are already processed (so we don't process them again)
	if ignore_already_processed_subjects:

		# remove already processed subjects from the file array
		gt3x_files = remove_processed_subjects(gt3x_files)

	# remove unwanted log.bin and info.txt files
	if remove_unwanted_files:

		# find all log.bin files and delete them
		for f in glob2.glob(os.path.join(gt3x_folder, '**', 'log.bin')):
			delete_file(f)
		# find all info.txt files and delete them
		for f in glob2.glob(os.path.join(gt3x_folder, '**', 'info.txt')):
			delete_file(f)


	# if use_parallel is set to True, then use parallelization to process all files
	if use_parallel:

		logging.info('Processing in parallel (parallelization on)')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(process_gt3x_file)(f, i, len(gt3x_files)) for i, f in enumerate(gt3x_files))
		# execute tasks
		executor(tasks)

	else:
		# process files one by one
		for i, f in enumerate(gt3x_files):

			# call process_gt3x_file function, f = file name, and i = index of file, len(gt3x_files) = total number of gt3x files to be processed
			process_gt3x_file(f, i, len(gt3x_files))


def process_gt3x_file(f, i = 1, total = 1, hdf5_save_location = HDF5_SAVE, delete_zip_folder = True):
	"""
	Process .gt3x file
	- unzip into log.bin and info.txt
	- extract information from info.txt
	- extract information from log.bin
	- save data to hdf5 file

	Parameters
	----------
	f : string
		file location of the .gt3x file
	i : int (optional)
		index of file to be processed, is used to display a counter of the process. Default = 1. For example, processing 12/20
	total : int (optional)
		total number of files to be processed, is used to display a counter of the process. Default = 1. For example, processing 12/20
	hdf5_save_location : os.path
		folder location where to save the extracted acceleration data to.
	"""

	logging.debug('Processing GTX3 binary file: {} {}/{}'.format(f, i + 1, total))

	# unzip the raw .gt3x file: this will provide a log.bin and info.txt file
	# the save_location is a new folder with the same name as the .gt3x file
	log_bin, info_txt = unzip_gt3x_file(f, save_location = f.split('.')[0])

	# check if unzipping went ok
	if log_bin is not None:

		# print verbose
		logging.debug('log.bin location: {}'.format(log_bin))
		logging.debug('info.txt location: {}'.format(info_txt))

		# get info data from info file
		info_data = extract_info(info_txt)

		# check if subject name could be read from the binary file
		if info_data['Subject_Name'] is not "":

			# check if subject ID already processed
			if info_data['Subject_Name'] not in get_all_subjects_hdf5(hdf5_file = HDF5_SAVE):

				# retrieve log_data; i.e. accellerometer data and log_time; timestamps of acceleration data
				log_data, log_time = extract_log(log_bin, acceleration_scale = float(info_data['Acceleration_Scale']), sample_rate = int(info_data['Sample_Rate']))

				# check if log data is not None (with None something went wrong during reading of the binary file)
				if log_data is not None:

					# save log_data to HDF5 file
					save_data_to_group_hdf5(group = info_data['Subject_Name'], data = log_data, data_name = 'log', meta_data = info_data, overwrite = True, hdf5_file = hdf5_save_location)
					
					# save log_time data to HDF file
					save_data_to_group_hdf5(group = info_data['Subject_Name'], data = log_time, data_name = 'time', meta_data = info_data, overwrite = True, hdf5_file = hdf5_save_location)
				
				else:
					logging.error('Unable to convert .gt3x file: {} (subject {})'.format(f, info_data['Subject_Name']))
			else:
				logging.info('Subject name already defined as group in HDF5 file: {}, skipping..'.format(info_data['Subject_Name']))
		else:
			logging.error("Unable to read subject from info.txt file, skipping file: {}".format(f))
	else:
		logging.error("Error unzipping file: {}".format(f))
		
	# delete the created zip folder
	if delete_zip_folder:
		delete_directory(f.split('.')[0])

	# print time and memory
	set_end(tic, process)


def remove_processed_subjects(gt3x_files):
	"""
	Remove files that are already processed
	It extract the subject ID from the file name location, and checkes if that ID is already part of the HDF5 file

	Parameters
	---------
	gt3x_files : list
		list of file locations of the raw gt3x file

	Returns
	---------
	gt3x_files : list
		filtered list of gt3x files (removed file locations that are already processed)
	"""

	# read already processed subject IDs
	processed_subjects = get_all_subjects_hdf5(hdf5_file = HDF5_SAVE)
		
	# here we extract the 8 digit subject ID from the file name and see if it has already been processes, if so, then we don't want to include it
	gt3x_files = [file for file in gt3x_files if re.search(r'[0-9]{8}', file).group(0) not in processed_subjects]
	
	# return the filtered files
	return gt3x_files


if __name__ == "__main__":

	# start timer and memory counter
	tic, process, logging = set_start()

	# start batch processing all the gt3x files
	batch_process_gt3x_files(use_parallel = True)
	
	# print time and memory
	set_end(tic, process)



	# # Code to use for matlab

	# import matlab.engine

	# # start matlabe engine (use when wanting to extract activity from matlab scripts)
	# matlab = matlab.engine.start_matlab()
	# matlab.addpath('matlab',nargout=0)

	# # use below if you want to execute the matlab script
	# log_data, log_time = matlab.ExtractBin(log_bin, nargout=2)

	# # convert to numpy (only when using matlab code)
	# log_data = np.array(log_data._data).reshape(log_data.size, order='F')
	# log_time = np.array(log_time._data, dtype='int').reshape(log_time.size, order='F')

	# info_data = matlab.ExtractInfo(info_txt, nargout=1)
