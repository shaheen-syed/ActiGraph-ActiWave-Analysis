# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import os
import re
import sqlite3

"""
	IMPORT FUNCTIONS
"""
from functions.helper_functions import set_start, set_end, read_directory, read_csv

class DB():

	def __init__(self, db_file):
		"""
		Constructor
		"""

		# db connection
		self.conn = None
		# db file location
		self.db_file = db_file

		# create the connection
		self.create_connection()

	def create_connection(self):
		"""
		Setup Sqlite3 connection
		"""

		try:
			self.conn = sqlite3.connect(self.db_file)
		except Exception as e:
			logging.error('Unable to create DB connection: {}'.format(e))
			exit(1)

	def execute_query(self, query):
		"""
		Execute query to DB

		Paramaters
		----------
		query : string
			SQL query to be executed
		"""

		# setup the cursor
		cur = self.conn.cursor()
		# execute query
		cur.execute(query)
		# return results
		return cur.fetchall()
	
	def read_table(self, table):
		"""
		Read all records from specified table name

		Paramaters
		----------
		table : string
			table name
		"""

		# setup the cursor
		cur = self.conn.cursor()
		# execute query
		cur.execute("SELECT * FROM {}".format(table))
		# return results
		return cur.fetchall()

	def execute_commit(self):
		"""
		Commit changes to db file
		"""

		self.conn.commit()

	
if __name__ == '__main__':

	tic, process, logging = set_start()

	# set to true if unprocessed files need to be deleted
	delete_unprocessed = True

	# location of agd files
	agd_folder = os.path.join(os.sep, 'Volumes', 'Lacie', 'AGD')

	# location of the CSV file with subject meta data
	meta_data_file = os.path.join(os.sep, 'Volumes', 'LaCie', 'AGD_TEST', 'vekt.csv')

	# convert CSV file to dictionary so we have access to each of the meta data by subject ID (skip the first row as it represents the column headers)
	meta_data = {x[4] : {'age' : x[0], 'sex' : 'Male' if x[1] == '1' else 'Female', 'height' : x[2], 'weight' : x[3]} for x in read_csv(meta_data_file)[1:]}

	# read all the .agd files
	F = [f for f in read_directory(agd_folder) if f[-4:] == '.agd']

	# keep track of unprocessed files
	no_meta_data = []
	incomplete_meta_data = []

	# process each .agd file, read content, and metadata and save back
	for i, f in enumerate(F):

		# extract subject from filename
		try:
			subject = re.search(r'[0-9]{8}', f)[0]
		except Exception as e:
			logging.error('Unable to extract subject ID from file name, skipping...')
			no_meta_data.append(f)
			continue
		
		logging.info('{style} Processing subject {}, file {}/{} {style}'.format(subject, i, len(F), style = '='*10))

		# setup the database connection
		db = DB(f)

		# get meta data from subject from dictionary (keep in mind the unpack order here)
		age, sex, height, weight = meta_data[subject].values()

		logging.info('Found age: {}, sex: {}, height: {}, weight: {}'.format(age, sex, height, weight))

		# check if all have valid data
		if any(x == '' for x in [age, sex, height, weight]):
			logging.error('Missing values in meta data, skipping...')
			incomplete_meta_data.append(f)
			continue

		"""
		Update age, sex, height and weight (mass)
		
		Example record
		(23, 'sex', 'Male'), 
		(24, 'height', '173.5'), 
		(25, 'mass', '73.8'), 
		(26, 'age', '34'), 
		"""
		db.execute_query("UPDATE settings SET settingValue = {} Where settingName = 'age'".format(age))
		db.execute_query("UPDATE settings SET settingValue = '{}' Where settingName = 'sex'".format(sex))
		db.execute_query("UPDATE settings SET settingValue = {} Where settingName = 'height'".format(height))
		db.execute_query("UPDATE settings SET settingValue = {} Where settingName = 'mass'".format(weight))

		# commit update
		db.execute_commit()


	# print out unprocessed agd files
	if len(no_meta_data) > 0:
		logging.info('Files without meta data (no subject mapping)')
		for i in no_meta_data: print(i)
	if len(incomplete_meta_data) > 0:	
		logging.info('Files without incomplete meta data')
		for i in incomplete_meta_data: print(i)

	if delete_unprocessed:
		for i in no_meta_data: os.remove(i)
		for i in incomplete_meta_data: os.remove(i)
	
	set_end(tic, process)
