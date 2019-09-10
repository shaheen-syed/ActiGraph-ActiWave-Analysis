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

def choi_2011_calculate_non_wear_time(data, time, activity_threshold = 0, min_period_len = 90, spike_tolerance = 2,  min_window_len = 30, use_vector_magnitude = False, print_output = False):
	"""	
	Estimate non-wear time based on Choi 2011 paper:

	Med Sci Sports Exerc. 2011 Feb;43(2):357-64. doi: 10.1249/MSS.0b013e3181ed61a3.
	Validation of accelerometer wear and nonwear time classification algorithm.
	Choi L1, Liu Z, Matthews CE, Buchowski MS.

	Description from the paper:
	1-min time intervals with consecutive zero counts for at least 90-min time window (window 1), allowing a short time intervals with nonzero counts lasting up to 2 min (allowance interval) 
	if no counts are detected during both the 30 min (window 2) of upstream and downstream from that interval; any nonzero counts except the allowed short interval are considered as wearing

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
	min_window_len : int (optional)
		minimum length of upstream or downstream time window (referred to as window2 in the paper) for consecutive zero counts required before and after the artifactual movement interval to be considered a nonwear time interval.
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
	non_wear_vector = np.ones((len(data),1), dtype = np.int16)

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
	# second window validation
	window_2_invalid = False
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
			window_2_invalid = False
			cnt_non_zero = 0

		# the non-wear period starts with a zero count
		if paxinten == 0 and start == False:
			
			# assign the starting minute of non-wear
			strt_nw = paxn
			# set start boolean to true so we know that we started the period
			start = True

		# only do something when the non-wear period has started
		if start:

			# keep track of the number of minutes with intensity that is not a 'zero' count
			if paxinten > 0:
				
				# increase the spike counter
				cnt_non_zero +=1

			# when there is a non-zero count, check the upstream and downstream window for counts
			# only when the upstream and downstream window have zero counts, then it is a valid non wear sequence
			if paxinten > 0:

				# check upstream window if there are counts, note that we skip the count right after the spike, since we allow for 2 minutes of spikes
				upstream = data[paxn + spike_tolerance: paxn + min_window_len + 1]

				# check if upstream has non zero counts, if so, then the window is invalid
				if (upstream > 0).sum() > 0:
					window_2_invalid = True

				# check downstream window if there are counts, again, we skip the count right before since we allow for 2 minutes of spikes
				downstream = data[paxn - min_window_len if paxn - min_window_len > 0 else 0: paxn - 1]

				# check if downstream has non zero counts, if so, then the window is invalid
				if (downstream > 0).sum() > 0:
					window_2_invalid = True

				# if the second window is invalid, we need to reset the sequence for the next run
				if window_2_invalid:
					reset = True

			# reset counter of value is zero again
			if paxinten == 0:
				cnt_non_zero = 0

			# the sequence ends when there are 3 consecutive spikes, or an invalid second window (upstream or downstream), or the last value of the sequence	
			if cnt_non_zero == 3 or window_2_invalid or paxn == len(data -1):
				
				# define the end of the period
				end_nw = paxn

				# check if the sequence is sufficient in length
				if len(data[strt_nw:end_nw]) < min_period_len:
					# lenght is not sufficient, so reset values in next run
					reset = True
				else:
					# length of sequence is sufficient, set stopped to True so we save the sequence start and end later on
					stopped = True

			# if stopped is True, the sequence stopped and is valid to include in the ranges
			if stopped:
				# add ranges start and end non wear time
				ranges.append([strt_nw, end_nw])


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