# encoding:utf-8

"""
	IMPORT PACKAGES
"""
import numpy as np
import logging

"""
	IMPORTED FUNCTIONS
"""
from functions.helper_functions import calculate_vector_magnitude

def troiano_2007_calculate_non_wear_time(data, time, activity_threshold = 0, min_period_len = 60, spike_tolerance = 2, spike_stoplevel = 100, use_vector_magnitude = False, print_output = False):
	"""
	Troiano 2007 non-wear algorithm
		detects non wear time from 60s epoch counts
		Nonwear was defined by an interval of at least 60 consecutive minutes of zero activity intensity counts, with allowance for 1–2 min of counts between 0 and 100
	Paper:
		Physical Activity in the United States Measured by Accelerometer
	DOI:
		10.1249/mss.0b013e31815a51b3

	Decription from original Troiano SAS CODE
	* A non-wear period starts at a minute with the intensity count of zero. Minutes with intensity count=0 or*;
	* up to 2 consecutive minutes with intensity counts between 1 and 100 are considered to be valid non-wear *;
	* minutes. A non-wear period is established when the specified length of consecutive non-wear minutes is  *;
	* reached. The non-wear period stops when any of the following conditions is met:                         *;
	*  - one minute with intensity count >100                                                                 *;
	*  - one minute with a missing intensity count                                                            *;
	*  - 3 consecutive minutes with intensity counts between 1 and 100                                        *;
	*  - the last minute of the day   


	Parameters
	------------
	data: np.array((n_samples, 3 axes))
		numpy array with 60s epoch data for axis1, axis2, and axis3 (respectively X,Y, and Z axis)
	time : np.array((n_samples, 1 axis))
		numpy array with timestamps for each epoch, note that 1 epoch is 60s
	activity_threshold : int (optional)
		The activity threshold is the value of the count that is considered "zero", since we are searching for a sequence of zero counts. Default threshold is 0
	min_period_len : int (optional)
		The minimum length of the consecutive zeros that can be considered valid non wear time. Default value is 60 (since we have 60s epoch data, this equals 60 mins)
	spike_tolerance : int (optional)
		Any count that is above the activity threshold is considered a spike. The tolerence defines the number of spikes that are acceptable within a sequence of zeros. The default is 2, meaning that we allow for 2 spikes in the data, i.e. aritifical movement
	spike_tolerenance_consecutive: Boolean (optional)
		Defines if spikes within the data need to be concecutive, thus following each other, or that the spikes can be anywhere within the non-wear sequence. Default is True, meaning that, if the tolerence is 2, two spikes need to follow each other.
	spike_stoplevel : int (Optional)
		any activity above the spike stoplevel end the non wear sequence, default to 100.
	use_vector_magnitude: Boolean (optional)
		if set to true, then use the vector magniturde of X,Y, and Z axis, otherwise, use X-axis only. Default False
	print_output : Boolean (optional)
		if set to True, then print the output of the non wear sequence, start index, end index, duration, start time, end time and epoch values. Default is False

	Returns
	---------
	non_wear_vector : np.array((n_samples, 1))
		numpy array with non wear time encoded as 0, and wear time encoded as 1.
	"""

	# check if data contains at least min_period_len of data
	if len(data) < min_period_len:
		logging.error('Epoch data contains {} samples, which is less than the {} minimum required samples'.format(len(data), min_period_len))

	# create non wear vector as numpy array with ones. now we only need to add the zeros which are the non-wear time segments
	non_wear_vector = np.ones((len(data),1), dtype = np.uint8)

	"""
		ADJUST THE COUNTS IF NECESSARY
	"""

	# if use vector magnitude is set to True, then calculate the vector magnitude of axis 1, 2, and 3, which are X, Y, and Z
	if use_vector_magnitude:
		# calculate vectore
		data = calculate_vector_magnitude(data, minus_one = False, round_negative_to_zero = False)
	else:
		# if not set to true, then use axis 1, which is the X-axis, located at index 0
		data = data[:,0]

	"""
		VARIABLES USED TO KEEP TRACK OF NON WEAR PERIODS
	"""

	# indicator for resetting and starting over
	reset = False
	# indicator for stopping the non-wear period
	stopped = False
	# indicator for starting to count the non-wear period
	start = False
	# starting minute for the non-wear period
	strt_nw = 0
	# ending minute for the non-wear period
	end_nw = 0
	# counter for the number of minutes with intensity between 1 and 100
	cnt_non_zero = 0
	# keep track of non wear sequences
	ranges = []

	"""
		FIND NON WEAR PERIODS IN DATA
	"""

	# loop over the data
	for paxn in range(0, len(data)):

		# get the value
		paxinten = data[paxn]

		# reset counters if reset or stopped
		if reset or stopped:	
			
			strt_nw = 0
			end_nw = 0
			start = False
			reset = False
			stopped = False
			cnt_non_zero = 0

		# the non-wear period starts with a zero count
		if paxinten <= activity_threshold and start == False:
			# assign the starting minute of non-wear
			strt_nw = paxn
			# set start boolean to true so we know that we started the period
			start = True

		# only do something when the non-wear period has started
		if start:

			# keep track of the number of minutes with intensity between 1-100
			if activity_threshold < paxinten <= spike_stoplevel:
				# increase the spike counter
				cnt_non_zero +=1

			# before reaching the 3 consecutive minutes of 1-100 intensity, if encounter one minute with zero intensity, reset the counter
			# this means that 0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 0, 10, 10, 0, 0, 0 is valid
			# and 0, 0, 0, 10, 10, 10, is invalid because we have 3 consecutive spikes between 0-100
			if paxinten <= activity_threshold:
				cnt_non_zero = 0
		
			# A non-wear period ends with 3 consecutive minutes of 1-100 intensity, or one minute with >100 intensity or with the last sample of the data
			if paxinten > spike_stoplevel or cnt_non_zero > spike_tolerance or paxn == len(data) -1:
				
				# define the end of the period
				end_nw = paxn

				# check if the sequence is sufficient in length
				if len(data[strt_nw:end_nw]) < min_period_len:
					# length is not sufficient
					reset = True
				else:
					# length is sufficient, save the period
					ranges.append([strt_nw, end_nw])
					stopped = True

	# convert ranges into non-wear sequence vector
	for row in ranges:

		# if set to True, then print output to console/log
		if print_output:
			logging.debug('start index: {}, end index: {}, duration : {}'.format(row[0], row[1], row[1] - row[0]))
			logging.debug('start time: {}, end time: {}'.format(time[row[0]], time[row[1]]))
			logging.debug('Epoch values \n{}'.format(data[row[0]:row[1]].T))
		
		# set the non wear vector according to start and end
		non_wear_vector[row[0]:row[1]] = 0

	return non_wear_vector