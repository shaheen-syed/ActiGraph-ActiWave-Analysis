# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import sys
import logging
import resampy # to resample frequency
import numpy as np
from scipy import signal
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

def apply_butterworth_filter(data, n, wn, btype, hz):
	"""
	Butterworth digital and analog filter design.

	Design an Nth-order digital or analog Butterworth filter and return the filter coefficients.

	See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

	Parameters
	----------
	data: np.array(n_samples, axes)
		numpy array with acceleration data (each column represents an axis)
	n : int
		the order of the filter
	wn: np.array(1,2)
		A scalar or length-2 sequence giving the critical frequencies. For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
		For digital filters, Wn are in the same units as fs. By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in half-cycles / sample.)
		For analog filters, Wn is an angular frequency (e.g. rad/s).
	btype: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
		The type of filter. Default is ‘lowpass’.
	hz 	: float
		The sampling frequency of the digital system.

	Returns
	--------
	data_filtered : np.array(n_samples, axes)
		numpy array with filtered acceleration data
	"""

	logging.info('Start {}'.format(sys._getframe().f_code.co_name))

	# create new numpy array to populate with the filtered data
	data_filtered = np.empty(data.shape)

	# loop over each acceleration axis
	for i in range(data.shape[1]):

		# get filter values
		B2, A2 = signal.butter(n, wn / (hz / 2), btype = btype)

		# apply filter and add to empty array
		data_filtered[:,i] = signal.filtfilt(B2, A2, data[:,i])

	return data_filtered


def resample_acceleration(data, from_hz, to_hz, use_parallel = False, num_jobs = cpu_count()):
	"""
	Resample acceleration data to different frequency. For example, convert 100hz data to 30hz data.
	Enables upsampling (from lower to higher frequency), or downsampling (from higher to lower frequency)

	Uses the resampy python module.
	see: https://github.com/bmcfee/resampy

	Used in this paper:
	Smith, Julius O. Digital Audio Resampling Home Page Center for Computer Research in Music and Acoustics (CCRMA), Stanford University, 2015-02-23. Web published at http://ccrma.stanford.edu/~jos/resample/.

	Parameters
	----------
	data: np.array
		numpy array with acceleration data, can be more than one dimension
	from_hz: int
		original sample frequency of the data (this is usually the frequency the device was set to during initialization)
	to_hz : int
		the sampling frequency to convert to.

	Returns
	--------
	new_data : np.array
		new numpy array with resampled acceleration data
	"""

	logging.info('Start {}'.format(sys._getframe().f_code.co_name))

	# calculate number of 1 sec samples (note that hz is the frequency per second)
	num_seconds = len(data) // from_hz

	# calculate number of new samples required when data is resampled
	num_samples = num_seconds * to_hz

	# get number of axes in the data. These are the columns of the array (so if we have xyz then this is 3)
	axes = data.shape[1]

	# create new empty array that we can populate with the resampled data
	new_data = np.zeros((num_samples, axes))

	if use_parallel:
		
		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')

		# create tasks so we can execute them in parallel
		tasks = (delayed(resample)(data[:,i], from_hz, to_hz, i) for i in range(axes))

		# execute tasks in parallel. It returns the resampled columns and column index i
		for i, column_data in executor(tasks):

			# add column data to correct column index
			new_data[:,i] = column_data
		
		# finished and return new data
		return new_data

	else:
		# loop over each of the columns of the original data, resample, and then add to the new_data array
		for i in range(axes):

			_, new_data[:,i] = resample(data[:,i], from_hz, to_hz, i)

		return new_data


"""
	Internal Helper Function
"""
def resample(data, from_hz, to_hz, index):
	"""
	Resample data from_hz to to_hz

	data: np.array(n_samples, 1)
		numpy array with single column
	from_hz: int
		original sample frequency of the data (this is usually the frequency the device was set to during initialization)
	to_hz : int
		the sampling frequency to convert to.
	index : int
		column index. Is used when use_parallel is set to True and the index is then used to know which column index is being returned. 

	Returns
	-------
	index : int
		column index, see above
	new_data : np.array(n_samples, 1)
		new numpy array with resampled acceleration data
	"""

	logging.debug('Processing axis {}'.format(index))

	return index, resampy.resample(data, from_hz, to_hz)