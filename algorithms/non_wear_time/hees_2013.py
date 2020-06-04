# encoding:utf-8

"""
	IMPORT PACKAGES
"""
import numpy as np
import logging

def hees_2013_calculate_non_wear_time(data, hz = 100, min_non_wear_time_window = 60, window_overlap = 15, std_mg_threshold = 3.0, std_min_num_axes = 2 , value_range_mg_threshold = 50.0, value_range_min_num_axes = 2):
	"""
	Estimation of non-wear time periods based on Hees 2013 paper

	Estimation of Daily Energy Expenditure in Pregnant and Non-Pregnant Women Using a Wrist-Worn Tri-Axial Accelerometer
	Vincent T. van Hees  , Frida Renström , Antony Wright, Anna Gradmark, Michael Catt, Kong Y. Chen, Marie Löf, Les Bluck, Jeremy Pomeroy, Nicholas J. Wareham, Ulf Ekelund, Søren Brage, Paul W. Franks
	Published: July 29, 2011https://doi.org/10.1371/journal.pone.0022922

	Accelerometer non-wear time was estimated on the basis of the standard deviation and the value range of each accelerometer axis, calculated for consecutive blocks of 30 minutes. 
	A block was classified as non-wear time if the standard deviation was less than 3.0 mg (1 mg = 0.00981 m·s−2) for at least two out of the three axes or if the value range, for 
	at least two out of three axes, was less than 50 mg.

	Parameters
	----------
	data: np.array(n_samples, axes)
		numpy array with acceleration data in g values. Each column represent a different axis, normally ordered YXZ
	hz: int (optional)
		sample frequency in hertz. Indicates the number of samples per 1 second. Default to 100 for 100hz. The sample frequency is necessary to 
		know how many samples there are in a specific window. So let's say we have a window of 15 minutes, then there are hz * 60 * 15 samples
	min_non_wear_time_window : int (optional)
		minimum window length in minutes to be classified as non-wear time
	window_overlap : int (optional)
		basically the sliding window that progresses over the acceleration data. Defaults to 15 minutes.
	std_mg_threshold : float (optional)
		standard deviation threshold in mg. Acceleration axes values below or equal this threshold can be considered non-wear time. Defaults to 3.0g. 
		Note that within the code we convert mg to g.
	std_min_num_axes : int (optional) 
		minimum numer of axes used to check if acceleration values are below the std_mg_threshold value. Defaults to 2 axes; meaning that at least 2 
		axes need to have values below a threshold value to be considered non wear time
	value_range_mg_threshold : float (optional)
		value range threshold value in mg. If the range of values within a window is below this threshold (meaning that there is very little change 
		in acceleration over time) then this can be considered non wear time. Default to 50 mg. Note that within the code we convert mg to g
	value_range_min_num_axes : int (optional)
		minimum numer of axes used to check if acceleration values range are below the value_range_mg_threshold value. Defaults to 2 axes; meaning that at least 2 axes need to have a value range below a threshold value to be considered non wear time

	Returns
	---------
	non_wear_vector : np.array((n_samples, 1))
		numpy array with non wear time encoded as 0, and wear time encoded as 1.
	"""

	# number of data samples in 1 minute
	num_samples_per_min = hz * 60

	# define the correct number of samples for the window and window overlap
	min_non_wear_time_window *= num_samples_per_min
	window_overlap *= num_samples_per_min

	# convert the standard deviation threshold from mg to g
	std_mg_threshold /= 1000
	# convert the value range threshold from mg to g
	value_range_mg_threshold /= 1000

	# new array to record non-wear time. Convention is 0 = non-wear time, and 1 = wear time. Since we create a new array filled with ones, we only have to 
	# deal with non-wear time (0), since everything else is already encoded as wear-time (1)
	non_wear_vector = np.ones((data.shape[0], 1), dtype = 'uint8')

	# loop over the data, start from the beginning with a step size of window overlap
	for i in range(0, len(data), window_overlap):

		# define the start of the sequence
		start = i
		# define the end of the sequence
		end = i + min_non_wear_time_window

		# slice the data from start to end
		subset_data = data[start:end]

		# check if the data sequence has been exhausted, meaning that there are no full windows left in the data sequence (this happens at the end of the sequence)
		# comment out if you want to use all the data
		if len(subset_data) < min_non_wear_time_window:
			break

		# calculate the standard deviation of each column (YXZ)
		std = np.std(subset_data, axis=0)

		# check if the standard deviation is below the threshold, and if the number of axes the standard deviation is below equals the std_min_num_axes threshold
		if (std < std_mg_threshold).sum() >= std_min_num_axes:

			# at least 'std_min_num_axes' are below the standard deviation threshold of 'std_min_num_axes', now set this subset of the data to 0 which will 
			# record it as non-wear time. Note that the full 'new_wear_vector' is pre-populated with all ones, so we only have to set the non-wear time to zero
			non_wear_vector[start:end] = 0

		# calculate the value range (difference between the min and max) (here the point-to-point numpy method is used) for each column
		value_range = np.ptp(subset_data, axis = 0)

		# check if the value range, for at least 'value_range_min_num_axes' (e.g. 2) out of three axes, was less than 'value_range_mg_threshold' (e.g. 50) mg
		if (value_range < value_range_mg_threshold).sum() >= value_range_min_num_axes:

			# set the non wear vector to non-wear time for the start to end slice of the data
			# Note that the full array starts with all ones, we only have to set the non-wear time to zero
			non_wear_vector[start:end] = 0

	return non_wear_vector