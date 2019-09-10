# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import logging
import time
import h5py
import os
from functions.helper_functions import get_random_number_between


def get_hdf5_file():
	"""
	Return the file path of the HDF5 file
	
	Returns
	-------
	hdf5_file : string
		file location of the HDF5 file (where we store the raw acceleration data)
	"""

	# return os.path.join(os.sep, 'Users', 'shaheensyed', 'Actigraph', 'raw_acc_data.hdf5')
	return os.path.join(os.sep, 'Volumes', 'LaCie_server', 'ACTIGRAPH_TU7.hdf5')


def get_all_subjects_hdf5(hdf5_file = None, filter_on = None):
	"""
	Get all the subjects from the hdf5 file where the raw data is stored
	These are the keys of the groups, as we created a group for each subject to separate the data

	Parameters
	----------
	hdf5_file : string (optional)
		location of the hdf5 file. If not given, then we read it from the function get_hdf5_file
	filter_on : string (optional)
		name of the dataset to filter on. Thus returns groups that contain the filter_on dataset. Default is None, so all groups are returned within the HDF5 file

	Returns
	--------
	hf.keys() : list
		list of group keys, which are the names of the subjects extracted from the gtx3 files
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()

	# check if file exists
	try:

		if os.path.exists(hdf5_file):

			# open the hdf5 file
			with h5py.File(hdf5_file, 'r') as hf:

				# check if filter_on contains a value, if so, we want return only subjects (i.e. groups) that contain a certain dataset
				if filter_on is None:

					# return the keys of the HDF5 file. Here keys are subject IDs
					# note that we can't return hf.keys since they are view like objects, that's why we have to convert to list first
					return list(hf.keys())

				else:

					# create empty list
					subjects = []

					# check for each subject if filter_on dataset exists
					for subject in hf.keys():

						# check if filter_on dataset is part of the group keys
						if filter_on in hf[subject].keys():
							
							# subject contains the filter_on dataset, append to list so we can return it later
							subjects.append(subject)

					return subjects	
					

		else:
			logging.warning('HDF5 file does not exist: {}'.format(hdf5_file))
			
			# return empty list
			return []

	except IOError:
	
		logging.warning('Could not read HDF5 file, possibly already open, retrying')
		
		# random sleep between seconds
		sleep = get_random_number_between(1, 5)
		time.sleep(sleep)
	
		# call function again
		return get_all_subjects_hdf5(hdf5_file)
	
	except Exception as e:
		logging.warning('Error reading subjects from HDF5 file: {}'.format(e))
		exit(1)


def create_group_in_hdf5_file(group, hdf5_file = None):
	"""
	Create group in HDF5 file

	Parameters
	----------
	group : string
		name of the group that should be created
	hdf5_file : string (optional)
		location of the hdf5 file. If not given, then we read it from the function get_hdf5_file
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()

	try:

		# store in HDF5: make sure you are in append mode 'a', when in write mode 'w', the file will be recreated
		with h5py.File(hdf5_file, 'a') as hf:

			# create group as subject name
			hf.create_group(group)
			
			logging.info('Succesfully create group {}'.format(group))
	except IOError:

		# random sleep between seconds
		sleep = get_random_number_between(1, 5)
		# logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
		time.sleep(sleep)

		# call function again because hdf5 file does not allow for multiple write sessions
		return create_group_in_hdf5_file(group, hdf5_file)

	except Exception as e:
		logging.warning('Error creating group in HDF5 file: {}'.format(e))
		exit(1)


def delete_group(group_name, hdf5_file = None):
	"""
	Delete group from HDF5 file

	Parameters
	----------
	group_name : string
		the name of the group to be deleted
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()	

	# check if file exists
	if os.path.exists(hdf5_file):
		try:
			with h5py.File(hdf5_file, 'a') as hf:

				# check if subject has its own group
				if group_name in hf.keys():

					# remove the group
					del hf[group_name]

				else:
					logging.error('Group {} not present in HDF5 file: {}'.format(group_name, hdf5_file))
					return None

		except IOError:

			# random sleep between seconds
			sleep = get_random_number_between(1, 5)
			# logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
			time.sleep(sleep)

			# call function again because hdf5 file does not allow for multiple write sessions
			return delete_group(group_name = group_name, hdf5_file = hdf5_file)

		except Exception as e:
			
			logging.error('Error deleting group {}: {}'.format(group_name, e))
			exit(1)
	else:
		logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
		exit(1)


def delete_dataset_from_group(group_name, dataset, hdf5_file = None):
	"""
	Delete a dataset from a group within the HDF5 file

	Parameters
	----------
	group_name : string
		the name of the group
	dataset : string
		name of the dataset that needs to be deleted
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()	

	# check if file exists
	if os.path.exists(hdf5_file):

		try:

			with h5py.File(hdf5_file, 'a') as hf:

				# check if subject has its own group
				if group_name in list(hf.keys()):

					if dataset in list(hf[group_name].keys()):

						# remove the group
						del hf[group_name][dataset]
					else:
						logging.warning('Dataset {} not present in group {}'.format(dataset, group_name))
						return None

				else:
					logging.error('Group {} not present in HDF5 file: {}'.format(group_name, hdf5_file))
					return None

		except IOError:

			# random sleep between seconds
			sleep = get_random_number_between(1, 3)
			# logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
			time.sleep(sleep)

			# call function again because hdf5 file does not allow for multiple write sessions
			return delete_dataset_from_group(group_name, dataset, hdf5_file)

		except Exception as e:
			
			logging.error('Error deleting group {}: {}'.format(group_name, e))
			exit(1)

	else:
		logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
		exit(1)


def read_dataset_from_group(group_name, dataset, hdf5_file = None, start_slice = None, end_slice = None, stride = 1):
	"""
	Read dataset from group of HDF5 file

	Parameters
	-----------
	group_name : string
		HDF5 group name
	dataset : string
		HDF5 dataset name
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function
	start_slice : int (optional)
		start slice of the data
	end_slice : int (optional)
		end slice of the  data
	stride : int (optional)
		stride of the data (i.e. skipping rows)

	Returns
	----------
	data : np.array
		numpy array of the dataset
	"""

	logging.debug('Reading group: {}, dataset: {}, hdf5_file: {}'.format(group_name, dataset, hdf5_file))

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()	

	try:
		# check if file exists
		if os.path.exists(hdf5_file):

			with h5py.File(hdf5_file, 'r') as hf:

				# check if subject has its own group
				if group_name in list(hf.keys()):

					# get the group
					group = hf[group_name]

					# get the dataset from the group
					if dataset in group.keys():

						# get the data
						return group[dataset][start_slice:end_slice:stride]

					else:
						logging.error('Dataset {} not part of group {}'.format(dataset, group_name))
						return None

				else:
					logging.error('Group {} not present in HDF5 file: {}'.format(group_name, hdf5_file))
					return None
		
		else:
			logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
			exit(1)
	
	except IOError:

		# random sleep between seconds
		sleep = get_random_number_between(1, 5)
		# logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
		time.sleep(sleep)

		# call function again because hdf5 file does not allow for multiple write sessions
		return read_dataset_from_group(group_name, dataset, hdf5_file, start_slice, end_slice, stride)
	
	except Exception as e:
		logging.error('Error reading dataset {} from group {}: {}'.format(dataset, group_name, e))
		exit()


def save_data_to_group_hdf5(group, data, data_name, meta_data = None, overwrite = False, create_group_if_not_exists = True, hdf5_file = None):
	"""
	Save data as a dataset in a group

	Parameters
	---------
	group : string
		The name of the group where the data needs to be stored. Can be subject name for example
	data : numpy.array
		Data that needs to be stored.
	data_name : string
		Name of the dataset. This is basically the key within the group
	meta_data : dictionary (optional)
		meta data to save with the group, should be of type dictionary. For instance, header information or subject information
	overwrite : Boolean (optional)
		If set to True then we overwrite the current dataset if present
	create_group_if_not_exists = Boolean (optional)
		create group in hdf5 file if not exists
	hdf5_file : string (optional)
		location of the hdf5 file. If not given, then we read it from the function get_hdf5_file
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()

	try:

		# store in HDF5: make sure you are in append mode 'a', when in write mode 'w', the file will be recreated
		with h5py.File(hdf5_file, 'a') as hf:

			# check if group exists
			group_exists = True if hf.get(group) is not None else False

			# check if group needs to be created if not exist
			if create_group_if_not_exists:
				# only create group is not exist already
				if not group_exists:
					# create group
					create_group_in_hdf5_file(group = group, hdf5_file = hdf5_file)
			else:
				# don't create group if not exist but if group not exists, issue warning, because otherwise we can't add data to the group
				if not group_exists:
					logging.warning('Could not add data {} to group because group {} does not exist. Continue with setting the create_group_if_not_exists parameter to True.'.format(data_name, group))
					exit(1)

			# define group as variable
			grp = hf[group]

			# check if overwrite is set to true. If so, then we need to delete the dataset first
			if overwrite:
				
				# delete dataset
				if grp.get(data_name) is not None:
					del grp[data_name]

			# check if dataset already exists in group
			if data_name not in grp.keys() or overwrite == True:

				# store data in group
				grp.create_dataset(data_name, data = data)
				logging.info('Dataset {} saved in group {}'.format(data_name, group))

				# store meta data if present
				if meta_data is not None:

					# check if meta_data is a dictionary
					if isinstance(meta_data, dict):

						# add meta data to dataset
						for key, value in meta_data.items():
							grp[data_name].attrs[key] = value

						logging.info('Meta data saved to group: {} with name {}'.format(group, data_name))
					else:
						logging.error('Meta data is not of type dictionary. Received {} instead'.format(type(meta_data)))

			else:

				logging.warning('Dataset {} already exists in group {}. Consider setting overwrite = True if you want to overwrite the data.'.format(data_name, group))
			
	except IOError:

		# random sleep between seconds
		sleep = get_random_number_between(1, 5)
		logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
		time.sleep(sleep)

		# call function again because hdf5 file does not allow for multiple write sessions
		return save_data_to_group_hdf5(group, data, data_name, meta_data , overwrite, create_group_if_not_exists, hdf5_file)

	except Exception as e:
		logging.error('Error saving dataset {} to group {}: {}'.format(data_name, group, e))
		exit()



def save_multi_data_to_group_hdf5(group, data, data_name, meta_data = None, overwrite = False, create_group_if_not_exists = True, hdf5_file = None):
	"""
	Save list of data as a dataset in a group (this functions has the same functionality as save_data_to_group_hdf5 but it allows for list of data to be inserted)

	Parameters
	---------
	group : string
		The name of the group where the data needs to be stored. Can be subject name for example
	data : numpy.array
		Data that needs to be stored.
	data_name : string
		Name of the dataset. This is basically the key within the group
	meta_data : dictionary (optional)
		meta data to save with the group, should be of type dictionary. For instance, header information or subject information
	overwrite : Boolean (optional)
		If set to True then we overwrite the current dataset if present
	create_group_if_not_exists = Boolean (optional)
		create group in hdf5 file if not exists
	hdf5_file : string (optional)
		location of the hdf5 file. If not given, then we read it from the function get_hdf5_file
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()

	# checks to see if lists are of equal length
	if len(data) != len(data_name):
		logging.error('Size of data and data_name are not the same.')
		exit(1)

	try:

		# store in HDF5: make sure you are in append mode 'a', when in write mode 'w', the file will be recreated
		with h5py.File(hdf5_file, 'a') as hf:

			# check if group exists
			group_exists = True if hf.get(group) is not None else False

			# check if group needs to be created if not exist
			if create_group_if_not_exists:
				# only create group if not exist already
				if not group_exists:
					# create group
					create_group_in_hdf5_file(group = group, hdf5_file = hdf5_file)
			else:
				# don't create group if not exist but if group not exists, issue warning, because otherwise we can't add data to the group
				if not group_exists:
					logging.warning('Could not add data {} to group because group {} does not exist. Continue with setting the create_group_if_not_exists parameter to True.'.format(data_name, group))
					exit(1)

			# define group as variable
			grp = hf[group]

			# check if overwrite is set to true. If so, then we need to delete the dataset first
			if overwrite:
				
				# delete datasets
				for i in range(0, len(data)): 
					if grp.get(data_name[i]) is not None:
						del grp[data_name[i]]

			# check if dataset already exists in group
			for i in range(0, len(data)):
				
				if data_name[i] not in grp.keys() or overwrite == True:

					# store data in group
					grp.create_dataset(data_name[i], data = data[i])

					logging.info('Dataset {} saved in group {}'.format(data_name[i], group))

					# store meta data if present
					if meta_data is not None:

						# check if meta_data is a dictionary
						if meta_data[i] is not None and isinstance(meta_data[i], dict):

							# add meta data to dataset
							for key, value in meta_data[i].items():
								grp[data_name[i]].attrs[key] = value

							logging.info('Meta data saved to group: {} with name {}'.format(group, data_name[i]))
						else:
							logging.warning('Meta data is not of type dictionary. Received {} instead. Skiping..'.format(type(meta_data[i])))

				else:

					logging.warning('Dataset {} already exists in group {}. Consider setting overwrite = True if you want to overwrite the data.'.format(data_name[i], group))
			
	except IOError:

		# random sleep between seconds
		sleep = get_random_number_between(1, 2)
		logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
		time.sleep(sleep)

		# call function again because hdf5 file does not allow for multiple write sessions
		return save_multi_data_to_group_hdf5(group, data, data_name, meta_data , overwrite, create_group_if_not_exists, hdf5_file)

	except Exception as e:
		logging.error('Error saving datasets to group {}: {}'.format(group, e))
		exit()


def read_metadata_from_group_dataset(group_name, dataset, hdf5_file = None):
	"""
	Read metadata from a dataset from group of HDF5 file

	Parameters
	-----------
	group_name : string
		HDF5 group name
	dataset : string
		HDF5 dataset name
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function
	
	Returns
	----------
	metadata : dictionary
		dictionary of meta-data
	"""


	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()	

	# check if file exists
	if os.path.exists(hdf5_file):

		try:

			with h5py.File(hdf5_file, 'r') as hf:

				# check if subject has its own group
				if group_name in hf.keys():

					# get the group
					group = hf[group_name]

					# get the dataset from the group
					if dataset in group.keys():

						# get the data
						return dict(group[dataset].attrs)

					else:
						logging.error('Dataset {} not part of group {}'.format(dataset, group_name))
						return None

				else:
					logging.error('Group {} not present in HDF5 file: {}'.format(group_name, hdf5_file))
					return None
					
		except IOError:

			# random sleep between seconds
			sleep = get_random_number_between(1, 2)
			# logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
			time.sleep(sleep)

			# call function again because hdf5 file does not allow for multiple write sessions
			return read_metadata_from_group_dataset(group_name, dataset, hdf5_file)
	else:
		logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
		exit(1)


def read_metadata_from_group(group_name, hdf5_file = None):
	"""
	Read metadata from from group of HDF5 file

	Parameters
	-----------
	group_name : string
		HDF5 group name
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function
	
	Returns
	----------
	metadata : dictionary
		dictionary of meta-data
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()	

	# check if file exists
	if os.path.exists(hdf5_file):

		with h5py.File(hdf5_file, 'r') as hf:

			# check if subject has its own group
			if group_name in hf.keys():

				# get the group
				group = hf[group_name]

				# get the data
				return dict(group.attrs)

			else:
				logging.error('Group {} not present in HDF5 file: {}'.format(group, hdf5_file))
				return None
	else:
		logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
		exit(1)


def save_meta_data_to_group(group_name, meta_data, hdf5_file = None):
	"""
	Save meta data to group attributes (note that here we save a dictionary on the group level, thus not on the dataset level)

	Parameters
	-----------
	group_name : string
		HDF5 group name
	metadata : dictionary
		dictionary of meta-data
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()	

	try:
		# check if file exists
		if os.path.exists(hdf5_file):

			with h5py.File(hdf5_file, 'a') as hf:

				# check if subject has its own group
				if group_name in hf.keys():

					# define the group
					grp = hf[group_name]

					# check if meta_data is a dictionary
					if isinstance(meta_data, dict):

						# add meta data to dataset
						for key, value in meta_data.items():

							# add the key value pair to the group attributes
							grp.attrs[key] = value

						logging.info('Meta data saved to group: {}'.format(group_name))

					else:

						logging.error('Meta data is not of type dictionary. Received {} instead'.format(type(meta_data)))

				else:

					logging.error('Group {} not present in HDF5 file: {}'.format(group_name, hdf5_file))
					return None
		else:
			logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
			return None
		
	except IOError:

		# random sleep between seconds
		sleep = get_random_number_between(1, 5)
		# logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
		time.sleep(sleep)

		# call function again because hdf5 file does not allow for multiple write sessions
		return save_meta_data_to_group(group_name, meta_data, hdf5_file)

	except Exception as e:
		logging.error('Error saving meta data to group {}: {}'.format(group_name, e))
		exit()


def save_meta_data_to_group_dataset(group_name, dataset, meta_data, hdf5_file = None):
	"""
	Save meta data to group attributes (note that here we save a dictionary on the group level, thus not on the dataset level)

	Parameters
	-----------
	group_name : string
		HDF5 group name
	dataset : string
		name of the dataset (this is a dataset within a group)
	metadata : dictionary
		dictionary of meta-data
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function
	"""

	# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()	

	try:
		# check if file exists
		if os.path.exists(hdf5_file):

			with h5py.File(hdf5_file, 'a') as hf:

				# check if subject has its own group
				if group_name in hf.keys():

					# define the group
					grp = hf[group_name]

					# get the dataset from the group
					if dataset in grp.keys():

						# check if meta_data is a dictionary
						if isinstance(meta_data, dict):

							# add meta data to dataset
							for key, value in meta_data.items():

								# add the key value pair to the group attributes
								grp[dataset].attrs[key] = value

							logging.info('Meta data saved to group: {} and dataset: {}'.format(group_name, dataset))

						else:

							logging.error('Meta data is not of type dictionary. Received {} instead'.format(type(meta_data)))
							return None

					else:
						logging.error('Dataset {} not part of group {}'.format(dataset, group_name))
						return None
				else:

					logging.error('Group {} not present in HDF5 file: {}'.format(group_name, hdf5_file))
					return None
		else:
			logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
			return None
		
	except IOError:

		# random sleep between seconds
		sleep = get_random_number_between(1, 5)
		# logging.warning('HDF5 file currently open, sleeping {} seconds'.format(sleep))
		time.sleep(sleep)

		# call function again because hdf5 file does not allow for multiple write sessions
		return save_meta_data_to_group_dataset(group_name, dataset, meta_data, hdf5_file)

	except Exception as e:
		logging.error('Error saving meta data to group {} and dataset {}: {}'.format(group_name,dataset, e))
		exit()


def get_datasets_from_group(group_name, hdf5_file = None):
	"""
	Returns the datasets from a group

	Parameters
	-----------
	group_name : string
		HDF5 group name
	hdf5_file : string (optional)
		path of the HDF5 file. If not given, then read from get_hdf5_file function

	Returns
	---------
	datasets : hf[group_name].keys()
		the datasets that a part of a group
	"""

		# if the hdf5 file is not given, then we read it from the function get_hdf5_file 
	if hdf5_file is None:
		hdf5_file = get_hdf5_file()

	# check if file exists
	if os.path.exists(hdf5_file):

		with h5py.File(hdf5_file, 'r') as hf:

			# check if subject has its own group
			if group_name in hf.keys():

				# return the keys of the groupname
				return list(hf[group_name].keys())

			else:
				logging.error('Group {} not present in HDF5 file: {}'.format(group_name, hdf5_file))
				return None
	else:
		logging.error('HDF5 file does not exist: {}'.format(hdf5_file))
		exit(1)

