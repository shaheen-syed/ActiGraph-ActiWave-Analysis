# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import os
import logging
import re
import sys
import numpy as np
from datetime import datetime

"""
	IMPORT FUNCTIONS
"""
from functions.helper_functions import get_date_format_parser


def get_epoch_folder():
	"""
	Return the folder where the epoch data is stored
	
	Returns
	-------
	f : os.path
		folder location where the epoch data is stored
	"""

	return os.path.join(os.sep, 'Volumes', 'LaCie', 'Actigraph_epoch_10')


def parse_epoch_file(file, dtype = np.uint16):
	"""
	Parse the content of the epoch file
		- the header and return as a dictionary of values
		- the epoch data as a numpy array

	Parameters
	---------
	file : os.path
		location of the epoch data as csv file
	dtype : numpy data type (optional)
		the data type of the array where to store the epoch data, default to 16bit unsigned integer.

	Returns
	--------
	header : dictionary
		Contains all the header information from the CSV file (basically what Actilife puts there)
	data : numpy array
		Contains the epoch data per epoch time (e.g., 10 seconds)
	"""

	logging.info('Processing file: {}'.format(file))

	# define header labels for parsing
	header_labels = ['Serial Number', 'Start Time', 'Start Date', 'Epoch Period (hh:mm:ss)', 'Download Time', 'Download Date', 'Current Memory Address', 'Current Battery Voltage']

	# create empty header dictionary
	dic_header = {}

	# add file name to dictionary of header information
	dic_header['File Name'] = file

	# read in the file
	try:
		with open(file, mode='r') as f:

			# read line by line
			for i, l in enumerate(f):


				"""
					Here we read the lines of the csv file one by one. The first couple of lines contain the meta-data (see example output below). We basicially read the csv file 
					line by line until we reach the line that only contains ---------------. This line ends the meta data, and everything that comes after that is the epoch data.
					We use a regular expression to find that ----------- line, and if we do, we use next(f) to see the content of the next line. In the example below, there is nothing but
					epoch data, but sometimes there are column headers. Based on the content of that line, we know where the acceleration data is located, i.e. in what columns. This will then
					be used for np.loadtxt to load everything into a matrix.

					------------ Data File Created By ActiGraph wGT3XBT ActiLife v6.11.7 Firmware v1.3.0 date format dd.MM.yyyy Filter Normal -----------			
					Serial Number: xxxxx			
					Start Time 00:00:00			
					Start Date 04.07.2015			
					Epoch Period (hh:mm:ss) 00:00:10			
					Download Time 13:00:18			
					Download Date 07.08.2015			
					Current Memory Address: 0			
					Current Battery Voltage: 3	71     Mode = 13		
					--------------------------------------------------			
					50	127	279	0

					In the example above, the last line contains the epoch data with X, Y, Z, steps as column values.

				"""

				# read until the end of the header has been reached, this is a line with only ------
				if not re.match(r'-*\n', l):

					# check if the line is the header of the header (this is a raw string that contains usefull information that we parse here)
					if 'Data File Created' in l:

						# parse label value pairs from header of header line
						dic_header = parse_header_of_header_line(l, dic_header)

					# parse out the value of the header label
					for label in header_labels:
						dic_header = parse_single_epoch_header_line(l, label, dic_header)


				else:

					# here we have reached the end of the header parts, retrieve the content of the next line, this will determine how the epoch data is structured
					# keep in mind that we only look at the next line, the index is still on the current line so to say. So skiprows i + 1 is mimimally necessary to 
					# get the values after the line ---------------------------
					next_line = next(f)

					# convert the csv into a numpy array (unsigned 8 bits should be sufficient to store the values)
		
					# match variant 1
					if re.match(r'axis1,axis2,axis3,steps,subjectname',next_line):
						data = np.loadtxt(file, delimiter = ',', skiprows = i + 2, usecols = (0,1,2,3), dtype = dtype)
		
					# match variant 2
					elif re.match(r'axis1,axis2,axis3,steps',next_line):
						data = np.loadtxt(file, delimiter = ',', skiprows = i + 2, dtype = dtype)

					# match variant 3
					elif re.match(r'timestamp,filename,epochlengthinseconds,serialnumber,axis1,axis2,axis3,steps,lux,vectormagnitude,capsense,inclineoff,inclinestanding,inclinesitting,inclinelying,gender,subjectname', next_line):
						data = np.loadtxt(file, delimiter = ',', skiprows = i + 2, usecols = (4,5,6,7), dtype = dtype)

					# all other variants (there is no row with column headers so we only have the skip i + 1 ) (see also the example csv output in the large comment block above)
					else:	
						data = np.loadtxt(file, delimiter = ',', skiprows = i + 1, usecols = (0,1,2,3), dtype = dtype)

					# return the values
					return (dic_header, data)
	
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def convert_epoch_data(data, start_epoch_sec, end_epoch_sec):
	"""
	Converts numpy array with start_epoch_sec into accumulated data of end_epoch_sec
		- For example, it converts 10 sec epoch data into 60 seconds epoch data
		- Conversion is done by adding the values for the individual axes
		- So 60 seconds epoch data from 10 seconds epoch data sums the values for e.g. x-acceleration

	Parameters
	----------
	data: numpy array
		array containing the epoch data
	start_epoch_sec : int
		number of seconds the original epoch data has, for instance 10 for 10 seconds epoch data
	end_epoch_sec : int
		number of seconds the converted data needs to be, for instance 60 for 60 seconds epoch data.

	Returns
	-------
	new_data : np.array
		converted epoch data
	"""

	
	# check if conversion is possible
	if end_epoch_sec % start_epoch_sec != 0:
		logging.error('Conversion not possible. Consider changing end_epoch value')
		exit(1)

	# calculate epoch ratio betweeon end and start (in the example case, going from 10 to 60 sec epoch this would be 6). This is necessary to know how many rows we need to sum
	epoch_ratio = end_epoch_sec // start_epoch_sec

	# lenght of new array (we create the array first so we can populate it later, this is much faster then appending or combining arrays)
	l = len(data) // epoch_ratio

	# get number of columns of data
	num_columns = data.shape[1]
	
	# create the new arrayarray 
	new_data = np.zeros((l,num_columns), dtype = np.uint16)

	# create a new datarow for every row in the new_data array
	for i in range(l):

		# define the start slice
		start = i * epoch_ratio
		# define the end slice
		end = start + epoch_ratio
		# append the sum of columns of the data to the new_data array
		new_data[i,:] = np.sum(data[start:end], axis=0)

	# return new data 
	return new_data


def create_epoch_time_array(start_date, start_date_format, start_time, epoch_data_length, epoch_sec = 10):
	"""
	Create a time series numpy array from the start date and start time of the epoch file

	Parameters
	---------
	start_date : string
		start date in format dd.MM.yyyy or 
	start_date_format : string
		the format of the date, can be dd.MM.yyyy or M/d/yyyy
	start_time : string
		start time of the data HH:MM:SS
	epoch_data_length: int
		lenght of the orgiginal epoch data (so we know how long the time data needs to be)
	epoch_sec : int (optional)
		length of epoch data in seconds, default 10 sec epoch

	Returns
	--------
	time_data : numpy array of seconds
	"""

	# convert start date and start time to datetime format
	time_data = datetime.strptime('{} {}'.format(start_date, start_time), '{} %H:%M:%S'.format(get_date_format_parser(start_date_format)))
	
	# convert to datetime64 format numpy array
	time_data = np.asarray(time_data, dtype='datetime64[s]')

	# create a timedelta of epoch length intervals in the size of the epoch data length
	time_data = time_data + np.asarray(np.arange(0, epoch_data_length * epoch_sec, epoch_sec), dtype='timedelta64[s]')

	# flatten the time array
	time_data = time_data.flatten()

	return time_data


"""
	Internal Helper Function
"""
def parse_header_of_header_line(l, dic_header):
	"""
	The first line of the header contains some raw information that we need to parse
	Example is : ------------ Data File Created By ActiGraph wGT3XBT ActiLife v6.13.3 Firmware v1.6.1 date format dd.MM.yyyy Filter Normal Multiple Incline Limb: Undefined -----------

	Parameters
	---------
	l : string
		content of a single line
	dic_header: dictionary
		dictionary with label-value pairs

	Returns:
	dic_header : dictionary
		dictionary with the parsed label-value pairs
	"""

	try:
		# parse out the actilife version
		dic_header['ActiLife Version'] = re.findall(r'ActiLife v((?:[0-9]|\.)+)', l)[0]
		# parse out the actigraph firmware version
		dic_header['Actigraph Firmware'] = re.findall(r'Firmware v((?:[0-9]|\.)+)', l)[0]
		# parse out the filtering settings
		dic_header['Filter'] = re.findall(r'Filter (.*) -+', l)[0]
		# parse out the dataformat
		dic_header['Date Format'] = re.findall(r'date format (.*?) ', l)[0]
	except Exception as e:
		logging.error('Error parsing header of header line: {}'.format(e))
	finally:
		return dic_header


def parse_single_epoch_header_line(l, label, dic_header):
	"""
	Parse a single epoch header line to obtain the value for the label

	Parameters
	----------
	l : string
		content of a single line
	label: string
		label to parse the value for (for instance, Start Time)
	dic_header: dictionary
		dictionary with label-value pairs

	Returns
	----------
	dic_header : dictionary
		dictionary of label-value pairs so we have it available in the main function
	"""

	# try to parse the content of the line
	try:
		if re.match(label, l):

			# parse out the value of the predefined label
			value = re.findall(label + r':? (.*)\n', l)

			# check length
			if len(value) > 0:

				# add to dictioary
				dic_header[label] = value[0]

		return dic_header
	except Exception as e:
		logging.error('Error parsing single epoch line: {}'.format(e))
	finally:
		return dic_header