# encoding:utf-8

"""
	IMPORT PACKAGES
"""
import numpy as np
import logging


def hecht_2009_triaxial_calculate_non_wear_time(data, epoch_sec = 60, threshold = 5, time_interval_mins = 20):
	"""
	Calculate the non-wear time from a data array that contains the vector magnitude (VMU) according to Hecht 2009 algorithm

	Paper:
	COPD. 2009 Apr;6(2):121-9. doi: 10.1080/15412550902755044.
	Methodology for using long-term accelerometry monitoring to describe daily activity patterns in COPD.
	Hecht A1, Ma S, Porszasz J, Casaburi R; COPD Clinical Research Network.

	Parameters
	---------
	data : np.array((n_samples, 1))
		numpy array with n_samples rows and a single column that contains the vector magnitude
	epoch_sec : int (optional)
		How many seconds a single row contains (default 60 seconds) (means that each row contains epoch data of 60 seconds)
	threshold : int (optional)
		threshold value for Hecht non-wear time algorithm. Defaults to 5 VMU
	time_interval_mins : int (optional)
		time interval threshold for hecht non-wear time algorithm. Defaults to 20 min
	
	Returns
	---------
	non_wear_time_vector : np.array((n_samples, 1))
		numpy array with non wear time encoded as 0, and wear time encoded as 1.
	"""

	# calculate the sliding windows size based on incoming epoch data and time interval to determine non-wear-time
	time_window = time_interval_mins / (epoch_sec / 60.)

	# check of epoch seconds can fit in time interval of equal parts
	if time_window % 1 != 0:
		logging.error('Invalid epoch {} (sec) or time interval {} (mins). Time window ratio {}'.format(epoch_sec, time_interval_mins, time_window))
		exit(1)

	# set time_window to integer
	time_window = int(time_window)

	# define new numpy array where we store the non-wear values: 0 for non-wear and 1 for wear time. We initiate by setting the vector to ones so we only have to update it when we identify non-wear time
	non_wear_time_vector = np.ones((len(data), 1), dtype = np.uint8)

	# loop over each data row to determine if it is wear or non-wear time (note that we don't have to worry about index out of bounds, python will truncate accordingly)
	for i in range(len(data)):

		# Is the VMU value greater than threshold (e.g. 5)
		if data[i] > threshold:


			# of the following time_interval_mins (e.g. 20 mins), do at least 2 have a VMU of greater than threshold (e.g. 5)
			if not check_following_time_interval_threshold(data, i, time_window, threshold):

				# of the proceeding time_interval_minutes do at least 2 have VMU of greater than 5
				if not check_preceding_time_interval_threshold(data, i, time_window, threshold):

					# this is considered non-wear time, update data of the right row
					non_wear_time_vector[i] = 0

		else:
			# VMU is NOT great than threshold
			if not check_following_time_interval_threshold(data, i,time_window, threshold):

				# this is considered non-wear time
				non_wear_time_vector[i] = 0

			else:
				if not check_preceding_time_interval_threshold(data, i, time_window, threshold):

					# this is considered non-wear time
					non_wear_time_vector[i] = 0

	return non_wear_time_vector


def check_following_time_interval_threshold(data, index, time_window, threshold, min_count = 2):

	"""
	Part of Hecht's (2009) non-wear time algorithm
	check following [time_window] and see if at least min_count >= threshold
	"""

	# define the start slice, it will be the index +1 or index -1 depending on the operator.
	start_slice = index +  1
	# define the end slice, it will be the start slice plus or minus (depending on the operator) the time windows
	end_slice = start_slice + time_window


	return ((data[start_slice:end_slice] > threshold).sum()) >= min_count


def check_preceding_time_interval_threshold(data, index, time_window, threshold, min_count = 2):

	"""
	Part of Hecht's (2009) non-wear time algorithm
	check preceding [time_window] and see if at least min_count >= threshold
	"""

	# define the start slice, it will be the index +1 or index -1 depending on the operator.
	start_slice = index - 1
	# define the end slice, it will be the start slice plus or minus (depending on the operator) the time windows
	end_slice = start_slice - time_window

	# we look backwards so we reverse the order of start and end
	return ((data[end_slice:start_slice] > threshold).sum()) >= min_count


