# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import logging
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.dates as dates
import matplotlib.dates as md
from matplotlib import colors
sns.set_style("whitegrid")
plt.rcParams['agg.path.chunksize'] = 10000

"""
	IMPORT FUNCTIONS
"""
from functions.helper_functions import create_directory, get_current_timestamp


def plot_process_sensititivity_analysis_wear_time(df, annot, fmt, plot_folder, n):

	# plot the heatmap
	ax = sns.heatmap(df, cmap = "bwr_r", annot = annot, vmin = -300, vmax = 300, square = True, fmt = fmt, annot_kws = {"size": 10}, linewidths = .5, cbar = False, xticklabels=True, yticklabels=True)

	# adjust the figure somewhat
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')

	plt.yticks(rotation = 0)
	plt.xticks(rotation = 90, ha = 'left')
	plt.xlabel('Threshold (VMU)', fontsize = '14')
	plt.ylabel('Time Interval (mins)', fontsize = '14')
	plt.title('Average non-wear time deviation in mins/day compared to default Hecht (2009) values (n={}, {} = p<0.05, {} = p<0.01, {} = p<0.001)'.format(n, '*', r'$\dag$', r'$\ddag$'), fontsize = '15', y=1.04)
	fig = ax.get_figure()
	# set the size of the figure
	fig.set_size_inches(30, 20)

	# make sure plot folder exists
	create_directory(plot_folder)
	# save figure to this location
	save_location = os.path.join(plot_folder, 'sensitivity-analysis-{}.pdf'.format(get_current_timestamp()))
	# save figure
	fig.savefig(save_location, bbox_inches='tight')
	# close plot environment
	plt.close()


def plot_raw_activity_data_by_day(data, plot_folder, subject, plot_non_wear_time):

	#matplotlib.font_manager._rebuild()


	# setting up the plot environment
	fig, axs = plt.subplots(len(data),1, figsize=(40, 25))
	axs = axs.ravel()

	# create a counter for the subplots
	i = 0

	# loop over the data (day by day)
	for _, group in data:

		# plot the YXZ columns
		for column_name in group.columns[0:3]:
			axs[i].plot(group[column_name], label = column_name)

		# if set to True, then also plot the non-wear time data
		if plot_non_wear_time:
			# extract only non-wear time from dataframe column
			non_wear_time = group['raw-non-wear-time'].loc[group['raw-non-wear-time'] == 0]
			# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
			non_wear_time = non_wear_time.iloc[::6000]
			# plot non wear as scatter
			axs[i].scatter( y = np.repeat(-5,len(non_wear_time)),  x = non_wear_time.index, c = 'red', s = 30)


		# define format of dates
		xfmt = md.DateFormatter('%H:%M')
		# define hours
		hours = md.HourLocator(interval = 1)

		# set the title as year - month - day
		axs[i].set_title('{} {}'.format(group.index[0].strftime('%Y-%m-%d'), group.index[0].strftime('%A')), fontsize = 22)
		# set the y axis limit
		axs[i].set_ylim((-5,5))
		# change the x-axis to show hours:
		axs[i].xaxis.set_major_locator(hours)
		# change the x-axis format
		axs[i].xaxis.set_major_formatter(xfmt)
		# change font size x acis
		axs[i].xaxis.set_tick_params(labelsize=20)
		# change font size y acis
		axs[i].yaxis.set_tick_params(labelsize=20)
		# set the legend
		axs[i].legend(loc="upper right", prop={'size': 20})
		# make sure the x-axis has no white space
		axs[i].margins(x = 0)

		# increase the counter
		i +=1
				
	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()


def plot_raw_vmu_activity_data_by_day(data, plot_folder, subject, plot_non_wear_time):

	matplotlib.rcParams['agg.path.chunksize'] = 10000

	# setting up the plot environment
	fig, axs = plt.subplots(3,1, figsize=(30, 10))
	axs = axs.ravel()

	# create a counter for the subplots
	i = 0
	# loop over the data (day by day)
	for _, group in data:

		# plot the YXZ columns
		for column_name in group.columns[0:3]:
			axs[i].plot(group[column_name], label = column_name)

		# plot the VMU data
		# axs[i].plot(group['RAW VMU'], label = 'RAW VMU')

		# if set to True, then also plot the non-wear time data
		if plot_non_wear_time:
			# extract only non-wear time from dataframe column
			non_wear_time = group['RAW NON WEAR TIME'].loc[group['RAW NON WEAR TIME'] == 0]
			# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
			non_wear_time = non_wear_time.iloc[::6000]
			# plot non wear as scatter
			axs[i].scatter( y = np.repeat(-5,len(non_wear_time)),  x = non_wear_time.index, c = 'red', s = 50)


		# set the title as year - month - day
		axs[i].set_title('{} {}'.format(group.index[0].strftime('%Y-%m-%d'), group.index[0].strftime('%A')), fontsize = 22)
		# set the y axis limit
		axs[i].set_ylim((-5,5))
		# increase the counter
		i +=1

		if i == 3:
			break

		# formatting for all axes
	for ax in axs:
		# define format of dates
		xfmt = md.DateFormatter('%H:%M')
		# define hours
		hours = md.HourLocator(interval = 1)
		# change the x-axis to show hours:
		ax.xaxis.set_major_locator(hours)
		# change the x-axis format
		ax.xaxis.set_major_formatter(xfmt)
		# change font size x acis
		ax.xaxis.set_tick_params(labelsize=20)
		# change font size y acis
		ax.yaxis.set_tick_params(labelsize=20)
		# set the legend
		ax.legend(loc="upper right", prop={'size': 20})
		# make sure the x-axis has no white space
		ax.margins(x = 0)

				
	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()

def plot_epoch_activity_data_by_day(data, plot_folder, subject, plot_counts, plot_vmu, plot_steps, plot_non_wear_time):

	# setting up the plot environment
	fig, axs = plt.subplots(len(data),1, figsize=(40, 25), sharey= True)
	axs = axs.ravel()
		
	# create a counter for the subplots
	i = 0

	# loop over the data (day by day)
	for _, group in data:
		
		# check if counts need to be plotted
		if plot_counts:
			# plot the XYZ columns
			for column_name in group.columns[0:3]:
				axs[i].plot(group[column_name], label = column_name)

		# check if VMU needs to be plotted
		if plot_vmu:
			axs[i].plot(group['VMU'], label = 'VMU')

		# check if steps need to be plotted
		if plot_steps:
			axs[i].plot(group['STEPS'], label = 'STEPS')

		# check if non wear time needs to be plotted
		if plot_non_wear_time:

			pass
			# # extract only non-wear time from dataframe column
			# non_wear_time = group['raw-non-wear-time'].loc[group['raw-non-wear-time'] == 0]
			# # make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
			# non_wear_time = non_wear_time.iloc[::6000]
			# # plot non wear as scatter
			# axs[i].scatter( y = np.repeat(-5,len(non_wear_time)),  x = non_wear_time.index, c = 'red', s = 30)

		# define format of dates
		xfmt = md.DateFormatter('%H:%M')
		# define hours
		hours = md.HourLocator(interval = 1)

		# set the title as year - month - day
		axs[i].set_title('{} {}'.format(group.index[0].strftime('%Y-%m-%d'), group.index[0].strftime('%A')), fontsize = 22)
		# change the x-axis to show hours:
		axs[i].xaxis.set_major_locator(hours)
		# change the x-axis format
		axs[i].xaxis.set_major_formatter(xfmt)
		# change font size x acis
		axs[i].xaxis.set_tick_params(labelsize=20)
		# change font size y acis
		axs[i].yaxis.set_tick_params(labelsize=20)
		# set the legend
		axs[i].legend(loc="upper right", prop={'size': 20})
		# make sure the x-axis has no white space
		axs[i].margins(x = 0)

		# increase the counter
		i +=1
				
	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()


def plot_raw_and_epoch_activity_data_by_day(data, plot_folder, subject, plot_raw_non_wear_time, plot_epoch_non_wear_time):



	# setting up the plot environment
	fig, axs = plt.subplots(len(data) * 2, 1, figsize=(40, 40))
	axs = axs.ravel()
		

	# define the start axis index for the raw data
	i = 0
	# loop over the data (day by day)
	for _, group in data:

		# plot the YXZ raw data columns
		for column_name in group.columns[0:3]:
			axs[i].plot(group[column_name], label = column_name)

		# if set to True, then also plot the raw non-wear time data
		if plot_raw_non_wear_time:
			# extract only non-wear time from dataframe column
			non_wear_time = group['RAW NON-WEAR-TIME'].loc[group['RAW NON-WEAR-TIME'] == 0]
			# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
			non_wear_time = non_wear_time.iloc[::6000]
			# plot non wear as scatter
			axs[i].scatter( y = np.repeat(-5,len(non_wear_time)),  x = non_wear_time.index, c = 'red', s = 60)

		# set the title as year - month - day
		axs[i].set_title('{} {}'.format(group.index[0].strftime('%Y-%m-%d'), group.index[0].strftime('%A')), fontsize = 22)
		# set the y axis limit
		axs[i].set_ylim((-5,5))

		# increase the counter by 2 because we want to 
		i += 2


	# define the start axis for the epoch data
	i = 1
	# get the max value of VMY epoch so we know what the y scale should be
	epoch_y_max = int(round(max([group['VMU - 60 EPOCH'].max() for name, group in data]),-3))
	
	# loop over the data (day by day)
	for name, group in data:
	
		# plot epoch 10S VMU data
		axs[i].plot(group['VMU - 10 EPOCH'].dropna(), label = 'VMU - 10 EPOCH')

		# plot epoch 60S VMU data
		axs[i].plot(group['VMU - 60 EPOCH'].dropna(), label = 'VMU - 60 EPOCH')

		# if set to True, then also plot the epoch non-wear time data
		if plot_epoch_non_wear_time:

			# extract only non-wear time from dataframe column
			epoch_non_wear_time = group['60 EPOCH NON-WEAR-TIME'].loc[group['60 EPOCH NON-WEAR-TIME'] == 0]
			
			# plot non wear as scatter
			axs[i].scatter( y = np.repeat(-300,len(epoch_non_wear_time)),  x = epoch_non_wear_time.index, c = 'blue', s = 30)

		# set the title as year - month - day
		axs[i].set_title('{} {}'.format(group.index[0].strftime('%Y-%m-%d'), group.index[0].strftime('%A')), fontsize = 22)

		# set the y axis limit
		axs[i].set_ylim((-300,epoch_y_max))

		# increase the counter by 2 because we want to 
		i += 2

	# formatting for all axes
	for ax in axs:
		# define format of dates
		xfmt = md.DateFormatter('%H:%M')
		# define hours
		hours = md.HourLocator(interval = 1)
		# change the x-axis to show hours:
		ax.xaxis.set_major_locator(hours)
		# change the x-axis format
		ax.xaxis.set_major_formatter(xfmt)
		# change font size x acis
		ax.xaxis.set_tick_params(labelsize=20)
		# change font size y acis
		ax.yaxis.set_tick_params(labelsize=20)
		# set the legend
		ax.legend(loc="upper right", prop={'size': 20})
		# make sure the x-axis has no white space
		ax.margins(x = 0)
				
	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn

	plt.show()
	plt.close()




def plot_actiwave_data(actigraph_acc, actiwave_acc, actiwave_hr, actiwave_ecg, plot_folder, subject):

	"""
		plot actiwave data together with actigraph
		- actigraph acceleration
		- actiwave acceleration
		- actiwave heart rate
		- actiwave ecg
		- actigraph and actiwave VMU

		Parameters
		----------
		actigraph_acc : pd.DataFrame()
			pandas dataframe with Y, X, Z, VMU data from actigraph (100 hz)
		actiwave_acc : pd.DataFrame()
			pandas dataframe with Y, X, Z, VMU data from actiwave (32 hz)
		actiwave_hr : pd.DataFrame()
			pandas dataframe with estimated heart rate data (1 measurement per second)
		actiwave_ecg : pd.DataFrame()
			pandas dataframe with ecg data (128hz)
	"""


	# setting up the plot environment
	fig, axs = plt.subplots(5, 1, figsize=(50, 40))
	axs = axs.ravel()

	"""
		ACTIGRAPH ACCELERATION DATA
	"""

	# plot acceleration Y
	axs[0].plot(actigraph_acc['Y'], label = 'Y')
	# plot acceleration X
	axs[0].plot(actigraph_acc['X'], label = 'X')
	# plot acceleration Z
	axs[0].plot(actigraph_acc['Z'], label = 'Z')
	# set the title as year - month - day
	axs[0].set_title('{} {} {}'.format('ACTIGRAPH ACCELERATION', actigraph_acc.index[0].strftime('%Y-%m-%d'), actigraph_acc.index[0].strftime('%A')), fontsize = 24)
	# set the y axis limit
	axs[0].set_ylim((-5,5))


	"""
		ACTIWAVE ACCELERATION DATA
	"""

	# plot acceleration Y
	axs[1].plot(actiwave_acc['Y'], label = 'Y')
	# plot acceleration X
	axs[1].plot(actiwave_acc['X'], label = 'X')
	# plot acceleration Z
	axs[1].plot(actiwave_acc['Z'], label = 'Z')
	# set the y axis limit
	axs[1].set_ylim((-5,5))
	# set the title
	axs[1].set_title('{}'.format('ACTIWAVE ACCELERATION'), fontsize = 24)


	"""
		ESTIMATED HEART RATE DATA
	"""

	axs[2].plot(actiwave_hr, label='ESTIMATED HR')
	# set the y axis limit
	axs[2].set_ylim((0,220))
	# set the title
	axs[2].set_title('{}'.format('ACTIWAVE HEART RATE'), fontsize = 24)

	"""
		ECG DATA
	"""
	axs[3].plot(actiwave_ecg, label='ECG')
	# set the y axis limit
	axs[3].set_ylim((-4000,4000))
	# set the title
	axs[3].set_title('{}'.format('ACTIWAVE ECG'), fontsize = 24)


	"""
		VMU
	"""

	# plot VMU
	axs[4].plot(actiwave_acc['VMU'], label = 'VMU ACTIWAVE', alpha= .7)
	axs[4].plot(actigraph_acc['VMU'], label = 'VMU ACTIGRAPH', alpha= .7)
	axs[4].set_title('{}'.format('VMU'), fontsize = 24)

	"""
		STYLING THE PLOT
	"""
	for ax in axs:
		# define format of dates
		xfmt = md.DateFormatter('%H:%M')
		# define hours
		hours = md.HourLocator(interval = 1)
		# change the x-axis to show hours:
		ax.xaxis.set_major_locator(hours)
		# change the x-axis format
		ax.xaxis.set_major_formatter(xfmt)
		# change font size x acis
		ax.xaxis.set_tick_params(labelsize=20)
		# change font size y acis
		ax.yaxis.set_tick_params(labelsize=20)
		# set the legend
		ax.legend(loc="upper right", prop={'size': 20})
		# make sure the x-axis has no white space
		ax.margins(x = 0)


	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()


def plot_non_wear_data(actigraph_acc, actiwave_acc, actiwave_hr, plot_folder, subject, annotations):

	try:

		# setting up the plot environment
		fig, axs = plt.subplots(4, 1, figsize=(50, 40))
		axs = axs.ravel()

		"""
			ACTIGRAPH ACCELERATION DATA
		"""

		# plot acceleration Y
		axs[0].plot(actigraph_acc['Y'], label = 'Y')
		# plot acceleration X
		axs[0].plot(actigraph_acc['X'], label = 'X')
		# plot acceleration Z
		axs[0].plot(actigraph_acc['Z'], label = 'Z')
		# set the title as year - month - day
		axs[0].set_title('{} {} {}'.format('ACTIGRAPH ACCELERATION', actigraph_acc.index[0].strftime('%Y-%m-%d'), actigraph_acc.index[0].strftime('%A')), fontsize = 24)
		# set the y axis limit
		axs[0].set_ylim((-5,5))

		"""
			ACTIGRAPH NON WEAR TIME
		"""
		# extract only non-wear time from dataframe column
		non_wear_time = actigraph_acc['NON-WEAR'].loc[actigraph_acc['NON-WEAR'] == 0]
		# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
		non_wear_time = non_wear_time.iloc[::2000]
		# plot non wear as scatter
		axs[0].scatter( y = np.repeat(-4.9,len(non_wear_time)),  x = non_wear_time.index, c = 'red', s = 60)


		# extract only non-wear time from dataframe column
		final_non_wear_time = actigraph_acc['NON-WEAR-FINAL'].loc[actigraph_acc['NON-WEAR-FINAL'] == 0]
		# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
		final_non_wear_time = final_non_wear_time.iloc[::2000]
		# plot non wear as scatter
		axs[0].scatter( y = np.repeat(-4.5,len(final_non_wear_time)),  x = final_non_wear_time.index, c = 'green', s = 60)

		# plot annotations
		for an in annotations:
			axs[0].annotate(s = an[1], xy = (an[0], -4.), fontsize = 20, rotation = 90)

		"""
			ACTIWAVE ACCELERATION DATA
		"""

		# plot acceleration Y
		axs[1].plot(actiwave_acc['Y'], label = 'Y')
		# plot acceleration X
		axs[1].plot(actiwave_acc['X'], label = 'X')
		# plot acceleration Z
		axs[1].plot(actiwave_acc['Z'], label = 'Z')
		# set the y axis limit
		axs[1].set_ylim((-5,5))
		# set the title
		axs[1].set_title('{}'.format('ACTIWAVE ACCELERATION'), fontsize = 24)

		"""
			ACTIWAVE NON WEAR TIME
		"""
		# extract only non-wear time from dataframe column
		non_wear_time = actiwave_acc['NON-WEAR'].loc[actiwave_acc['NON-WEAR'] == 0]
		# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
		non_wear_time = non_wear_time.iloc[::1000]
		# plot non wear as scatter
		axs[1].scatter( y = np.repeat(-4.9,len(non_wear_time)),  x = non_wear_time.index, c = 'red', s = 60)

		"""
			VMU
		"""

		# plot VMU
		axs[2].plot(actiwave_acc['VMU ACTIWAVE'], label = 'VMU ACTIWAVE', alpha= .7)
		axs[2].plot(actigraph_acc['VMU ACTIGRAPH'], label = 'VMU ACTIGRAPH', alpha= .7)
		axs[2].set_title('{}'.format('VMU'), fontsize = 24)

		
		"""
			ESTIMATED HEART RATE DATA
		"""

		axs[3].plot(actiwave_hr, label='ESTIMATED HR')
		# set the y axis limit
		axs[3].set_ylim((0,220))
		# set the title
		axs[3].set_title('{}'.format('ACTIWAVE HEART RATE'), fontsize = 24)


		"""
			STYLING THE PLOT
		"""
		for ax in axs:
			# define format of dates
			xfmt = md.DateFormatter('%H:%M')
			# define hours
			hours = md.HourLocator(interval = 1)
			# change the x-axis to show hours:
			ax.xaxis.set_major_locator(hours)
			# change the x-axis format
			ax.xaxis.set_major_formatter(xfmt)
			# change font size x acis
			ax.xaxis.set_tick_params(labelsize=20)
			# change font size y acis
			ax.yaxis.set_tick_params(labelsize=20)
			# set the legend
			ax.legend(loc="upper right", prop={'size': 20})
			# make sure the x-axis has no white space
			ax.margins(x = 0)


		# crop white space
		fig.set_tight_layout(True)
		# create the plot folder if not exist already
		create_directory(plot_folder)
		# create the save location
		save_location = os.path.join(plot_folder, '{}.png'.format(subject))
		# save the figure
		fig.savefig(save_location, dpi=150)
		# close the figure environemtn
		plt.close()
	except Exception as e:
		logging.error('Failed to plot non wear time: {}'.format(e))


def plot_non_wear_algorithms(data, subject, plot_folder):

	"""
		Plot actigraph acceleration data and estimated non wear time

		Parameters
		----------
		data: pd.DataFrame
			dataframe with columns = ['Y', 'X', 'Z', 'EPOCH 60s VMU', 'EPOCH 60s COUNT','TRUE NON WEAR TIME', 'HECHT-3 NON WEAR TIME','TROIANO NON WEAR TIME']
		subject: string
			subject ID
		plot_folder: os.path
			location where the plot needs to be saved
	"""

	# setting up the plot environment
	fig, axs = plt.subplots(4, 1, figsize=(40, 30))
	axs = axs.ravel()

	"""
		ACTIGRAPH ACCELERATION DATA
	"""

	# plot acceleration Y
	axs[0].plot(data['ACTIGRAPH Y'], label = 'Y')
	# plot acceleration X
	axs[0].plot(data['ACTIGRAPH X'], label = 'X')
	# plot acceleration Z
	axs[0].plot(data['ACTIGRAPH Z'], label = 'Z')
	# set the title
	axs[0].set_title('{} ({}) - {} {}'.format('ACTIGRAPH ACCELERATION', subject, data.index[0].strftime('%Y-%m-%d'), data.index[0].strftime('%A')), fontsize = 24)
	# set the y axis limit
	axs[0].set_ylim((-5,5))


	"""
		ACTIGRAPH NON WEAR TIME
	"""

	# extract only non-wear time from dataframe column
	true_non_wear_time = data['TRUE NON WEAR TIME'].loc[data['TRUE NON WEAR TIME'] == 0]
	# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
	true_non_wear_time = true_non_wear_time.iloc[::100 * 60]
	# plot non wear as scatter
	axs[0].scatter( y = np.repeat(-4.9,len(true_non_wear_time)),  x = true_non_wear_time.index, c = 'red', s = 20)
	
	"""
		HECHT 3-AXES NON WEAR TIME
	"""

	# hecht 3 axes non wear time
	hecht_3_non_wear_time = data['HECHT-3 NON WEAR TIME'].loc[data['HECHT-3 NON WEAR TIME'] == 0]
	# plot non wear as scatter
	axs[0].scatter( y = np.repeat(-4.6,len(hecht_3_non_wear_time)),  x = hecht_3_non_wear_time.index, c = 'blue', s = 20)

	"""
		TROIANO NON WEAR TIME
	"""

	# troiano axes non wear time
	troiano_non_wear_time = data['TROIANO NON WEAR TIME'].loc[data['TROIANO NON WEAR TIME'] == 0]
	# plot non wear as scatter
	axs[0].scatter( y = np.repeat(-4.4,len(troiano_non_wear_time)),  x = troiano_non_wear_time.index, c = 'green', s = 20)

	"""
		CHOI NON WEAR TIME
	"""

	# troiano axes non wear time
	choi_non_wear_time = data['CHOI NON WEAR TIME'].loc[data['CHOI NON WEAR TIME'] == 0]
	# plot non wear as scatter
	axs[0].scatter( y = np.repeat(-4.2,len(choi_non_wear_time)),  x = choi_non_wear_time.index, c = 'orange', s = 20)


	"""
		HEES NON WEAR TIME
	"""

	# troiano axes non wear time
	hees_non_wear_time = data['HEES NON WEAR TIME'].loc[data['HEES NON WEAR TIME'] == 0]
	# plot non wear as scatter
	axs[0].scatter( y = np.repeat(-4.0,len(hees_non_wear_time)),  x = hees_non_wear_time.index, c = 'pink', s = 20)



	"""
		ACTIWAVE ACCELERATION DATA
	"""

	# plot acceleration Y
	axs[1].plot(data['ACTIWAVE Y'].dropna(), label = 'Y')
	# plot acceleration X
	axs[1].plot(data['ACTIWAVE X'].dropna(), label = 'X')
	# plot acceleration Z
	axs[1].plot(data['ACTIWAVE Z'].dropna(), label = 'Z')
	# set the y axis limit
	axs[1].set_ylim((-5,5))
	# set the title
	axs[1].set_title('{}'.format('ACTIWAVE ACCELERATION'), fontsize = 24)



	"""
		EPOCH VMU DATA
	"""
	# plot epoch 60 vmu data
	axs[2].plot(data['EPOCH 60s VMU'].dropna(), label = 'EPOCH 60s VMU')
	# plot below
	axs[2].scatter( y = np.repeat(-100,len(hecht_3_non_wear_time)),  x = hecht_3_non_wear_time.index, c = 'blue', s = 20)


	# allow plotting on alternative y axis
	# ax2 = axs[2].twinx()
	# plot epoch 6 count data

	# set label size of ticks on right y axis
	# ax2.yaxis.set_tick_params(labelsize=20)
	# set the legend
	#ax2.legend(loc="upper right", prop={'size': 20})

	"""
		COUNT COUNT DATA
	"""
	# axs[3].plot(data['EPOCH 60s COUNT'].dropna(), label = 'EPOCH 60s COUNT')
	# # plot troiano non wear as scatter
	# axs[3].scatter( y = np.repeat(-100 ,len(troiano_non_wear_time)),  x = troiano_non_wear_time.index, c = 'green', s = 20)
	# # plot choi non wear as scatter
	# axs[3].scatter( y = np.repeat(-150 ,len(choi_non_wear_time)),  x = choi_non_wear_time.index, c = 'orange', s = 20)

	"""
		ESTIMATED HEART RATE DATA
	"""

	axs[3].plot(data['ESTIMATED HR'].dropna(), label='ESTIMATED HR')
	# set the y axis limit
	axs[3].set_ylim((0,220))
	# set the title
	axs[3].set_title('{}'.format('ACTIWAVE HEART RATE'), fontsize = 24)




	# # get handles and labels for axs[1]
	# handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
	# handles, labels = axs[2].get_legend_handles_labels()
	# handles.append(handles_ax2[0])
	# labels.append(labels_ax2[0])

	# # adjust the line width
	# for line in axs[1].get_lines(): line.set_linewidth(4.0)

	# add handles back to axis
	# axs[2].legend(handles, labels, loc='upper left', prop={'size': 20})


	
	"""
		STYLING THE PLOT
	"""
	for ax in axs:
		# define format of dates
		xfmt = md.DateFormatter('%H:%M')
		# define hours
		hours = md.HourLocator(interval = 1)
		# change the x-axis to show hours:
		ax.xaxis.set_major_locator(hours)
		# change the x-axis format
		ax.xaxis.set_major_formatter(xfmt)
		# change font size x acis
		ax.xaxis.set_tick_params(labelsize=20)
		# change font size y acis
		ax.yaxis.set_tick_params(labelsize=20)
		# set the legend
		ax.legend(loc='upper left', prop={'size': 20})
		# make sure the x-axis has no white space
		ax.margins(x = 0)

	"""
		MANUAL ADDING LEGENDS
	"""

	# define axis to add manual labels to
	add_to_axes = [0,2,3]

	for ax in add_to_axes:
		# get handles and labels for axs[0]
		handles, labels = axs[ax].get_legend_handles_labels()
		# define manual handles
		if ax == 0:
			manual_handles = {'red' : 'TRUE NON WEAR TIME', 'blue' : 'HECHT-3 NON WEAR TIME', 'green' : 'TROIANO NON WEAR TIME', 'orange' : 'CHOI NON WEAR TIME', 'pink' : 'HEES NON WEAR TIME'}
		elif ax == 2:
			manual_handles = {'blue' : 'HECHT-3 NON WEAR TIME'}
		else:
			manual_handles = {'green' : 'TROIANO NON WEAR TIME'}

		# add manual handles and lables
		for key, value in manual_handles.items():
			# append handle
			handles.append(mpatches.Patch(color=key, label=value))
			# add label
			labels.append(value)
		# add handles back to axis 
		axs[ax].legend(handles, labels, loc='upper left', ncol=2, prop={'size': 20})


	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()


def plot_roc_curve(fpr, tpr, auc):


	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()



"""
	TESTING
"""

def plot_signal(subject, data1, data2):


	plot_folder = os.path.join('plots', 'test')

	# setting up the plot environment
	fig, axs = plt.subplots(2, 1, figsize=(50, 20))
	axs = axs.ravel()

	axs[0].plot(data1)
	axs[1].plot(data2)
	

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()






def plot_test_counts(x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, y, subject):

	plot_folder = os.path.join('plots', 'test')

	# setting up the plot environment
	fig, axs = plt.subplots(10, 1, figsize=(50, 20))
	axs = axs.ravel()

	axs[0].plot(x, label='x')
	axs[0].set_title('30hz Actigraph GT3X RAW X-axis')
	

	axs[1].plot(x_1, label='x_1')
	axs[1].set_title('1) alias filter 0.01 - 7Hz')

	axs[2].plot(x_2, label='x_2')
	axs[2].set_title('2) frequency band pass filter')

	axs[3].plot(x_3, label='x_3')
	axs[3].set_title('3) down sample to 10hz')


	axs[4].plot(x_4, label='x_4')
	axs[4].set_title('4) rectification (convert to absolute values)')


	axs[5].plot(x_5, label='x_5')
	axs[5].set_title('5) truncate 2.13g')

	axs[6].plot(x_6, label='x_6')
	axs[6].set_title('6) dead band filter')

	axs[7].plot(x_7, label='x_7')
	axs[7].set_ylim((0,200))
	axs[7].set_title('7) convert to 8 bit resolution')

	axs[8].plot(x_8, label='x_8')
	axs[8].set_ylim((0,200))
	axs[8].set_title('8) accumalate 10 consecutive samples into 1 sec epoch')

	axs[9].plot(y, label='y')
	axs[9].set_ylim((0,200))
	axs[9].set_title('Actilife 1s epoch counts')


	for ax in axs:
		# make sure the x-axis has no white space
		ax.margins(x = 0)


	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()


def plot_counts(acc_30hz_x, epoch_x, epoch_x_predicted, y_hat_metrics, y_hat_filter_method, y_hat_filter_metrics,  subject):

	# save location
	plot_folder = os.path.join('plots', 'test')

	# setting up the plot environment
	fig, axs = plt.subplots(4, 1, figsize=(50, 20))
	axs = axs.ravel()

	# axs[0].plot(acc_100hz_x, label='100hz actigraph raw')
	# axs[0].set_title('100hz Actigraph GT3X RAW X-axis')
	# axs[0].set_ylim((-2,2))
	

	axs[0].plot(acc_30hz_x, label='30hz actigraph raw')
	axs[0].set_title('30hz Actigraph GT3X RAW X-axis')
	axs[0].set_ylim((-2,2))

	axs[1].plot(epoch_x, label='True epoch counts')
	axs[1].set_title('ActiLife True epoch counts')
	axs[1].set_ylim((0,200))

	axs[2].plot(epoch_x_predicted, label='Predicted epoch counts - ' + ' '.join(['{}: {},'.format(key, value) for key, value in y_hat_metrics.items()]))
	axs[2].set_title('Predicted epoch counts')
	axs[2].set_ylim((0,200))


	axs[3].plot(y_hat_filter_method, label='Filter method epoch counts  - ' + ' '.join(['{}: {},'.format(key, value) for key, value in y_hat_filter_metrics.items()]))
	axs[3].set_title('Filter method epoch counts')
	axs[3].set_ylim((0,200))



	for ax in axs:
		# set the legend
		ax.legend(loc='upper left', prop={'size': 20})
		# make sure the x-axis has no white space
		ax.margins(x = 0)

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, '{}.png'.format(subject))
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()


def plot_classification_performance(spec, prec, f1, plot_folder = os.path.join('plots', 'test')):

	"""
	Plot classification performance of non wear algorithms to true non wear time

	Parameters
	----------
	spec: np.array(samples, 4)
		numpy array with specificity data
	prec: np.array(samples, 4)
		numpy array with precision data
	f1: np.array(samples, 4)
		numpy array with f1 data
	"""

	# setting up the plot environment
	fig, axs = plt.subplots(2, 2, figsize=(20, 20))
	axs = axs.ravel()

	# define the labels of the array columns
	col_to_label = {0 : 'Hecht', 1 : 'Troiano', 2 : 'Choi', 3 : 'Hees'}

	# plot specificty
	for column in range(spec.shape[1]):

		# plot each column of data with associated label
		axs[0].plot(spec[:,column], label = col_to_label[column])


	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, 'classification_performance.png')
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()
