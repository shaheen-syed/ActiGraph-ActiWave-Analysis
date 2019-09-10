# encoding:utf-8

"""
	IMPORT PACKAGES
"""
import os
import sys
import logging
import pyedflib
import json
import numpy as np


def get_actiwave_folder():
	"""
	Return the folder where the actiwave cardio data is stored
	
	Returns
	-------
	f : os.path
		folder location where the cardio data is stored
	"""

	return os.path.join(os.sep, 'Volumes', 'LaCie', 'Actiwave')


def read_edf_file(file):
	"""
	Read an EDF FILE

	pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.

	EDF means [European Data Format](http://www.edfplus.info/) and was firstly published [1992](http://www.sciencedirect.com/science/article/pii/0013469492900097). 
	In 2003, an improved version of the file protokoll named EDF+ has been published and can be found [here](http://www.sciencedirect.com/science/article/pii/0013469492900097).

	Parameters
	----------
	file : string
		the location where the edf file is stored

	Returns
	---------
	dic_data : dictionary
		dictionary with the key : signal label, value : data as a numpy array
	"""

	logging.debug('Reading EDF file: {}'.format(file))

	try:

		# create EDF reader from the file location
		edf = pyedflib.EdfReader(file)

		# extract number of signals in file
		num_signals = edf.signals_in_file

		# extract signal labels
		signal_labels = edf.getSignalLabels()

		# create dictionary that we can return
		dic_data = {}

		# loop over number of signals
		for i in range(num_signals):

			# read edf signal from file, this returns a numpy array, and we add it to the dictionary
			dic_data[signal_labels[i]] = edf.readSignal(i)

		# return the data
		return dic_data
	
	except Exception as e:

		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_edf_meta_data(file):
	"""
	Read meta data of the EDF file

	Parameters
	---------
	file : string
		file location of the edf file

	Returns:
	meta_data : dictionary
		dictionary with meta data
	"""

	try:
		# create EDF reader from the file location
		edf = pyedflib.EdfReader(file)

		# create empty dictionary for edf meta data
		meta_data = {}

		meta_data['Signals in File'] = edf.signals_in_file
		meta_data['File Duration Seconds'] = edf.file_duration
		meta_data['Start Datetime'] = str(edf.getStartdatetime())
		meta_data['Patient Additional'] = edf.getPatientAdditional()
		meta_data['Patient Code'] = edf.getPatientCode()
		meta_data['Gender'] = edf.getGender()
		meta_data['Birthdate'] = edf.getBirthdate()
		meta_data['Patient Name'] = edf.getPatientName()
		meta_data['Admin Code'] = edf.getAdmincode()
		meta_data['Technician'] = edf.getTechnician()
		meta_data['Equipment'] = edf.getEquipment()
		meta_data['Recording Additional'] = edf.getRecordingAdditional()
		meta_data['Datarecords in File'] = edf.datarecords_in_file
		meta_data['Annotations in File'] = edf.annotations_in_file

		"""
			create the annotations as a big string
		"""

		# first create an empty dictionary
		dic_annotations = {}


		# zip the 3 annotations columns, basically row values from columns 0, 1, and 2 belong together , and then loop over them to save it to the empty dictionary
		for i, row in enumerate(zip(edf.readAnnotations()[0],edf.readAnnotations()[1],edf.readAnnotations()[2])):

			"""
				Example rows:
				[i] (annotation[0], annotation[1], annotation[2])

				0 (1512.0, 163.0, 'Vertical')
				1 (1819.0, 126.0, 'Vertical')
				2 (1987.0, 128.0, 'Vertical')
				3 (2117.0, 94.0, 'Vertical')
				4 (2266.0, 114.0, 'Vertical')
				5 (2480.0, 476.0, 'Vertical')
				6 (3427.0, 85.0, 'Vertical')
				7 (4403.0, 69.0, 'Vertical')
				8 (4489.0, 120.0, 'Vertical')
				9 (4611.0, 76.0, 'Vertical')
				10 (4893.0, 81.0, 'Vertical')
				11 (5997.0, 159.0, 'Slouched at 50Â° rolled 10Â° to right')

			"""
			dic_annotations[i] = row


		# add to meta data as a string
		meta_data['Annotations'] = np.string_(json.dumps(dic_annotations))

		return meta_data
	
	except Exception as e:

		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_edf_channel_meta_data(file, channel):
	"""
	Read the meta data of the individual channel, where a channel could be the ECG signal, or the X acceleration, or Y acceleration.
	
	Parameters
	---------
	file : string
		file location of the edf file
	channel : int
		channel of the signal ( channel 0 = ECG, channel 1, X acceleration, channel 2, Y acceleration, channel 3, Z acceleration, channel 4, estimated HR (heart rate))

	Returns
	----------
	meta_data : dictionary
		dictionary with meta data of the channel
	"""

	try:

		# create EDF reader from the file location
		edf = pyedflib.EdfReader(file)

		# create empty dictionary
		channel_meta_data = {}

		# extract data for edf channel
		channel_meta_data['Label'] = edf.getLabel(channel)
		channel_meta_data['NSamples'] = edf.getNSamples()[channel]
		channel_meta_data['Physical Maximum'] = edf.getPhysicalMaximum(channel)
		channel_meta_data['Physical Minimum'] = edf.getPhysicalMinimum(channel)
		channel_meta_data['Digital Maximum'] = edf.getDigitalMaximum(channel)
		channel_meta_data['Digital Minimum'] = edf.getDigitalMinimum(channel)
		channel_meta_data['Physical Dimension'] = edf.getPhysicalDimension(channel)
		channel_meta_data['Prefilter'] = edf.getPrefilter(channel)
		channel_meta_data['Transducer'] = edf.getTransducer(channel)
		channel_meta_data['Sample Frequency'] = edf.getSampleFrequency(channel)

		return channel_meta_data

	except Exception as e:

		logging.error('Failed to read EDF channel meta data: {} {}'.format(file, e))
		return None


def create_actiwave_time_vector(start_datetime, length, hz):
	"""
	Create a time array based on a start time, the number of data samples, and the frequenzy (how many times per second we have data)
	
	Parameters
	---------
	start_datetime: datetime
		the start of the data as a datetime in seconds precision
	length: int
		number of data samples (necessary to know how the lenght of the return array)
	hz: int
		the sampling frequency or frequency of the measurements, could be 128hz, 100hz, 32hz, or any other value. This indicates how many measurements per seconds were recorded by the device

	Returns
	--------
	time_data: numpy array in nanoseconds
		the time vector in nanoseconds precision. This precision is necessary to make sure that the sampling frequency fits into equal and whole parts within a 1 sec window.	
	"""

	# calculate how many seconds of data we have (num samples / hz)
	length_sec = float(length) / float(hz)

	# check if number of seconds is a whole number
	if not length_sec.is_integer():
		logging.error('Actiwave time in seconds not a whole number: {}'.format(length_sec))
		exit(1)

	"""
		h	hour	+/- 1.0e15 years	[1.0e15 BC, 1.0e15 AD]
		m	minute	+/- 1.7e13 years	[1.7e13 BC, 1.7e13 AD]
		s	second	+/- 2.9e11 years	[2.9e11 BC, 2.9e11 AD]
		ms	millisecond	+/- 2.9e8 years	[ 2.9e8 BC, 2.9e8 AD]
		us	microsecond	+/- 2.9e5 years	[290301 BC, 294241 AD]
		ns	nanosecond	+/- 292 years	[ 1678 AD, 2262 AD]
		ps	picosecond	+/- 106 days	[ 1969 AD, 1970 AD]
		fs	femtosecond	+/- 2.6 hours	[ 1969 AD, 1970 AD]
		as	attosecond
	"""

	# declare how many nanoseconds in one second
	ns_in_sec = 1000000000

	# check if the sampling frequenzy can fit into equal parts within a nanosecond window
	if ns_in_sec % hz != 0:
		logging.error('Sampling frequenzy {} cannot be split into equal parts within a 1s window'.format(hz))
		exit(1)

	# calculate the step size of hz in 1s (so 100hz means 100 measurements in 1sec, so if we need to fill 1000ms then we need use a step size of 10)
	step_size = ns_in_sec / hz

	# convert the start datetime (seconds precision) to a numpy array of 64 bit also in seconds precision
	time_data = np.asarray(start_datetime, dtype='datetime64[s]')

	# convert the array to 64 bit nanoseconds (necessary to fit 128hz for instance) and add a time delta of a range nanoseconds in seconds * lenght_sec, with corresponding stepsize
	time_data = np.asarray(time_data, dtype='datetime64[ns]') + np.asarray(np.arange(0, ns_in_sec * length_sec,step_size), dtype='timedelta64[ns]')
	
	# # flatten the array 
	time_data = time_data.flatten()

	return time_data