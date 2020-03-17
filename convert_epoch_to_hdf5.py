# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import os
import glob2
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

"""
	IMPORTED FUNCTIONS
"""
from functions.epoch_functions import parse_epoch_file
from functions.helper_functions import set_start, set_end
from functions.hdf5_functions import save_data_to_group_hdf5

"""
	GLOBAL VARIABLES
"""
# folder location of the .csv 10 seconds epoch files
EPOCH_FOLDER = os.path.join(os.sep, 'Volumes', 'LaCie', 'Actigraph_epoch_1')
# folder location to store data in HDF5 format to
HDF5_SAVE = os.path.join(os.sep, 'Volumes', 'LaCie', 'ACTIGRAPH_TU7.hdf5')


def batch_process_epoch_files(epoch_sec, epoch_folder = EPOCH_FOLDER, use_parallel = False, num_jobs = cpu_count(), limit = None, skip_n = 0):
	"""
	Read CSV epoch files from disk and extract (1) header information, and (2) epoch data for XYZ and also the steps.

	Parameters
	------------
	epoch_sec : int
		number of seconds within a single epoch. Examples include 1 for 1 sec epochs, or 10 for 10s epochs
	epoch_folder : os.path()
		folder location of the 10 seconds epoch files
	use_parallel = Boolean (optional)
		Set to true of subjects need to be processed in parallel, this will execute much faster
	num_jobs = int (optional)
		if parallel is set to true, then this indicates have many jobs at the same time need to be executed. Default set to the number of CPU cores
	limit : int (optional)
		limit the number of subjects to be processed
	skipN : int (optional)
		skip first N subjects
	"""

	# get all the .csv 10 seconds epoch files from the folder location. We do this here because there might also be other types of files in the folder
	# we can also skip_n the first n files, or it is possible to limit the number of files to be processed, such for testing or if we only need, for example, 100 files
	epoch_files = glob2.glob(os.path.join(epoch_folder, '**', '*.csv'))[0 + skip_n:limit]

	# if use_parallel is set to True, then use parallelization to process all files
	if use_parallel:

		logging.info('Processing in parallel (parallelization on)')

		# because we need to save the data after the parallel processing, we can't process them all at one since the return values becomes too large, so we peform in batches
		for i in range(0, len(epoch_files), num_jobs):

			# define start and end slice (these are the batches)
			start_slice = i
			end_slice = i + num_jobs

			# use parallel processing to speed up processing time
			executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')

			# create tasks so we can execute them in parallel
			tasks = (delayed(parse_epoch_file)(file = f) for f in epoch_files[start_slice:end_slice])

			# execute tasks and process the return values
			for dic_header, data in executor(tasks):

				# parse out subject ID from file name (split on /, then take the last, then split on space, and take the first)
				subject = dic_header['File Name'].split('/')[-1].split(' ')[0]
				dic_header['Subject'] = subject

				# save header and data to HDF5 file
				save_data_to_group_hdf5(group = subject, data = data, data_name = 'epoch{}'.format(epoch_sec), meta_data = dic_header, overwrite = True, create_group_if_not_exists = True, hdf5_file = HDF5_SAVE)

			# verbose
			logging.debug('{style} Processed {}/{} {style}'.format(end_slice, len(epoch_files), style = '='*10))

	else:

		# process files one-by-one
		for i, f in enumerate(epoch_files):

			logging.debug('{style} Processing epoch file: {} {}/{} {style}'.format(f, i + 1, len(epoch_files), style = '='*10))

			# parse the content from the epoch csv file
			dic_header, data = parse_epoch_file(f)

			# parse out subject ID from file name (split on /, then take the last, then split on space, and take the first)
			subject = dic_header['File Name'].split('/')[-1].split(' ')[0]
			dic_header['Subject'] = subject
			
			# save header and data to HDF5 file
			save_data_to_group_hdf5(group = subject, data = data, data_name = 'epoch{}'.format(epoch_sec), meta_data = dic_header, overwrite = True, create_group_if_not_exists = True, hdf5_file = HDF5_SAVE)


if __name__ == "__main__":

	# start timer and memory counter
	tic, process, logging = set_start()

	"""
		IMPORTANT:
		the epoch_sec needs to match the global variable EPOCH_FOLDER
		for example, if epoch_sec is 10 (meaning 10 sec epoch) then we need to point EPOCH folder to the correct 10 second epoch folder, for example os.path.join(os.sep, 'Volumes', 'LaCie', 'Actigraph_epoch_10')
		if the epoch is 1s based, then change epoch_sec to 1, and point the EPOCH folder to the correct 1s folder.
	"""

	# batch process all the epoch files
	batch_process_epoch_files(epoch_sec = 1, use_parallel = True)
	
	# print time and memory
	set_end(tic, process)