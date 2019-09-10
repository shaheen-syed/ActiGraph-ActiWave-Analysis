# encoding:utf-8
"""
	IMPORT PACKAGES
"""
import numpy as np


def find_candidate_non_wear_segments_from_raw(acc_data, std_threshold, hz, min_segment_length = 1, sliding_window = 1):
	"""
	Find segements within the raw acceleration data that can potentially be non-wear time (finding the candidates)

	Parameters
	---------
	acc_data : np.array(samples, axes)
		numpy array with acceleration data (typically YXZ)
	std_threshold : int or float
		the standard deviation threshold in mg (milli-g)
	hz : int
		sample frequency of the acceleration data (could be 32hz or 100hz for example)
	min_segment_length : int (optional)
		minimum length of the segment to be candidate for non-wear time (default 1 minutes, so any shorter segments will not be considered non-wear time)
	hz : int (optional)
		sample frequency of the data (necessary so we know how many data samples we have in a second window)
	sliding_window : int (optional)
		sliding window in minutes that will go over the acceleration data to find candidate non-wear segments
	"""

	# adjust the sliding window to match the samples per second (this is encoded in the samplign frequency)
	sliding_window *= hz * 60
	# adjust the minimum segment lenght to reflect minutes
	min_segment_length*= hz * 60

	# define new non wear time vector that we initiale to all 1s, so we only have the change when we have non wear time as it is encoded as 0
	non_wear_vector = np.ones((len(acc_data), 1), dtype = np.uint8)
	non_wear_vector_final = np.ones((len(acc_data), 1), dtype = np.uint8)

	# loop over slices of the data
	for i in range(0,len(acc_data), sliding_window):

		# slice the data
		data = acc_data[i:i + sliding_window]
				
		# calculate the standard deviation of each column (YXZ)
		std = np.std(data, axis=0)

		# check if all of the standard deviations are below the standard deviation threshold
		if np.all(std <= std_threshold):

			# add the non-wear time encoding to the non-wear-vector for the correct time slices
			non_wear_vector[i:i+sliding_window] = 0

	# find all indexes of the numpy array that have been labeled non-wear time
	non_wear_indexes = np.where(non_wear_vector == 0)[0]

	# find the min and max of those ranges, and increase incrementally to find the edges of the non-wear time
	for row in find_consecutive_index_ranges(non_wear_indexes):

		# check if not empty
		if row.size != 0:

			# define the start and end of the index range
			start_slice, end_slice = np.min(row), np.max(row)

			# backwards search to find the edge of non-wear time vector
			start_slice = backward_search_non_wear_time(data = acc_data, start_slice = start_slice, end_slice = end_slice, std_max = std_threshold, hz = hz) 
			# forward search to find the edge of non-wear time vector
			end_slice = forward_search_non_wear_time(data = acc_data, start_slice = start_slice, end_slice = end_slice, std_max = std_threshold, hz = hz)

			# calculate the length of the slice (or segment)
			length_slice = end_slice - start_slice

			# minimum length of the non-wear time
			if length_slice >= min_segment_length:

				# update numpy array by setting the start and end of the slice to zero (this is a non-wear candidate)
				non_wear_vector_final[start_slice:end_slice] = 0

	# return non wear vector with 0= non-wear and 1 = wear
	return non_wear_vector_final


def find_consecutive_index_ranges(vector, increment = 1):
	"""
	Find ranges of consequetive indexes in numpy array

	Parameters
	---------
	data: numpy vector
		numpy vector of integer values
	increment: int (optional)
		difference between two values (typically 1)

	Returns
	-------
	indexes : list
		list of ranges, for instance [1,2,3,4],[8,9,10], [44]
	"""

	return np.split(vector, np.where(np.diff(vector) != increment)[0]+1)


def forward_search_non_wear_time(data, start_slice, end_slice, std_max, hz, time_step = 60):
	"""
	Increase the end_slice to obtain more non_wear_time (used when non-wear range has been found but due to window size, the actual non-wear time can be slightly larger)

	Parameters
	----------
	data: numpy array of time x 3 axis 
		raw log data
	start_slice: int
		start of known non-wear time range
	end_slice: int
		end of known non-wear time range
	std_max : int or float
		the standard deviation threshold in g
	time_step : int (optional)
		value to add (or subtract in the backwards search) to find more non-wear time
	"""

	# adjust time step on number of samples per time step window
	time_step *= hz

	# define the end of the range
	end_of_data = len(data)

	# Do-while loop
	while True:

		# define temporary end_slice variable with increase by step
		temp_end_slice = end_slice + time_step

		# check condition range still contains non-wear time
		if temp_end_slice <= end_of_data and np.all(np.std(data[start_slice:temp_end_slice], axis=0) <= std_max):
			
			# update the end_slice with the temp end slice value
			end_slice = temp_end_slice

		else:
			# here we have found that the additional time we added is not non-wear time anymore, stop and break from the loop by returning the updated slice
			return end_slice


def backward_search_non_wear_time(data, start_slice, end_slice, std_max, hz, time_step = 60):
	"""
	Decrease the start_slice to obtain more non_wear_time (used when non-wear range has been found but the actual non-wear time can be slightly larger, so here we try to find the boundaries)

	Parameters
	----------
	data: numpy array of time x 3 axis 
		raw log data
	start_slice: int
		start of known non-wear time range
	end_slice: int
		end of known non-wear time range
	std_max : int or float
		the standard deviation threshold in g
	time_step : int (optional)
		value to add (or subtract in the backwards search) to find more non-wear time
	"""

	# adjust time step on number of samples per time step window
	time_step *= hz

	# Do-while loop
	while True:

		# define temporary end_slice variable with increase by step
		temp_start_slice = start_slice - time_step

		# logging.debug('Decreasing temp_start_slice to: {}'.format(temp_start_slice))

		# check condition range still contains non-wear time
		if temp_start_slice >= 0 and np.all(np.std(data[temp_start_slice:end_slice], axis=0) <= std_max):
			
			# update the start slice with the new temp value
			start_slice = temp_start_slice

		else:
			# here we have found that the additional time we added is not non-wear time anymore, stop and break from the loop by returning the updated slice
			return start_slice