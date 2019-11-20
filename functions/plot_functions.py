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
from matplotlib.lines import Line2D
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
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
	for _, group in data:
	
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
	save_location = os.path.join(plot_folder, '{}_opt.png'.format(subject))
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




def plot_time_distribution(data, full_time_range, plot_folder = os.path.join('plots', 'test'), plot_name = 'time_distribution.pdf'):
	"""
	plot distribution over hours of the day for all subjects

	Parameters
	----------
	data : pd.DataFrame()
		pandas dataframe with ['subject frequency'], ['nw frequency'], and ['nw percentage']
	plot_foler : os.path (optional)
		folder to store figure in

	"""
	# import pandas as pd
	# # list with time stamps per minute/per hour
	# hour_min_range = pd.date_range('2017-01-01 00:00', '2017-01-01 23:59', freq = '1T')
	
	# setting up the plot environment
	fig, axs = plt.subplots(1, 3, figsize = (15, 5))
	axs = axs.ravel()
	
	# create the x labels
	x_labels = ['{}:{}'.format(str(x.hour).zfill(2), str(x.minute).zfill(2)) for x in full_time_range]
	x = np.arange(len(data.index))

	# plot settings
	width = .8
	linewidth = 0
	align = 'center'
	
	axs[0].bar(x, data['subject frequency'], width = width, align = align, color = '#1b9e77', linewidth = linewidth)
	axs[0].set_title('Distribution Activity data (frequency)')
	
	axs[1].bar(x, data['nw frequency'], width = width, align = align, color = '#d95f02', linewidth = linewidth)
	axs[1].set_title('Distribution non-wear data (frequency)')
	
	axs[2].bar(x, data['nw percentage'], width = width, align = align, color = '#7570b3', linewidth = linewidth)
	axs[2].set_title('Relative Distribution non-wear data (%)')

	# adjust all the axes
	for ax in axs:

		# set x labels (only take every hour)
		ax.set_xticklabels(x_labels[::60])
		# set x ticks
		ax.set_xticks(x[::6])
		# no grid lines
		ax.grid(False)
		# no padding on x axis
		ax.margins(x = 0)
		
		# rotate all the x tick labels
		for tick in ax.get_xticklabels():
			tick.set_rotation(45)

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, plot_name)
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()


def plot_grid_search(data, labels, annotations, plot_parameters, plot_name, plot_folder = os.path.join('plots', 'test2')):


	fig, axs = plt.subplots(plot_parameters['num_rows'], plot_parameters['num_columns'], figsize = plot_parameters['figsize'])
	axs = axs.ravel()

	# loop over data
	counter = 0
	for combination, df in data.items():

		# get the x and y label
		x_label, y_label = combination.split('_')
		
		x = df.columns
		y = df.index
		X, Y = np.meshgrid(x, y)
		Z = df.values
				
		cs = axs[counter].contour(X, Y, Z, levels = plot_parameters['levels'], cmap='magma_r', vmin = plot_parameters['vmin'], vmax= plot_parameters['vmax'],  linewidths=2)
		axs[counter].clabel(cs, inline = True, fontsize=12)

		# set axes title
		axs[counter].set_xlabel(labels[x_label])
		axs[counter].set_ylabel(labels[y_label])
		# remove grid lines
		axs[counter].grid(False)

		axs[counter].locator_params(integer=True)

		if plot_parameters['annotations']:
			# plot the a dot to indicate optimal parameter values
			dot = axs[counter].plot(annotations[x_label], annotations[y_label], 'ro', zorder = 10)
			# make sure the dot is on top of axes splines
			dot[0].set_clip_on(False)
		
			# colors = ['r', 'g']
			# lines = [Line2D([0], [0], markersize = 10, marker = 'o', color = 'w', markerfacecolor=c) for c in colors]
			# legend_labels = ['default', 'optimized']
			# leg = axs[0].legend(lines, legend_labels, frameon=False)
			# leg.set_alpha(1)
			# leg.set_zorder(102)
			# leg.get_frame().set_facecolor('w')
			# annotate
			# axs[counter].annotate('default', (default_x, default_y))
			# axs[counter].annotate('optimized', (top_x, top_y))

		counter +=1

	# remove plots if parameter is set
	if plot_parameters.get('remove_plots') is not None:
		[axs[x].set_visible(False) for x in plot_parameters['remove_plots']] 

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, plot_name)
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()

def plot_nw_scenarios(all_data, plot_folder = os.path.join('plots', 'paper'), plot_name = 'nw-scenarios.png'):

	# plt.style.use("bmh")
	fig, axs = plt.subplots(3, 3, figsize = (35,10))
	axs = axs.ravel()

	# define colors
	c = ['#1b9e77', '#d95f02', '#7570b3', '#66a61e']
	c = ['#84C8E4', '#6FE468', '#D04896', '#272A6F']
	# define the counter for the subplots
	cnt = 0
	for dic_data in all_data.values():

		# read data from dictionary
		data = dic_data['data']

		"""
			Actigraph
		"""

		# plot acceleration Y
		axs[0 + cnt].plot(data['ACTIGRAPH Y'], label = 'Y', color = c[0])
		# plot acceleration X
		axs[0 + cnt].plot(data['ACTIGRAPH X'], label = 'X', color = c[1])
		# plot acceleration Z
		axs[0 + cnt].plot(data['ACTIGRAPH Z'], label = 'Z', color = c[2])
		# set the y axis limit
		axs[0 + cnt].set_ylim((-4,4))
		# set inner title
		axs[0 + cnt].text(0.5 ,.9,'ActiGraph accelerometer', horizontalalignment='center', transform = axs[0 + cnt].transAxes, fontdict = {'size' : 20})
		
		# extract candidate segment
		candidate_segment = data['CANDIDATE NW EPISODE'].loc[data['CANDIDATE NW EPISODE'] == 0]
		# make smaller by taking only each minute of data (we don't need 100 values per second here for plotting)
		candidate_segment = candidate_segment.iloc[::10 * 60]
		
		# plot candidate non-wear segment (episode)
		if cnt == 2:
			# for illustrative purposes, only show the last candidate episode
			axs[0 + cnt].scatter( y = np.repeat(-2,len(candidate_segment[70:])),  x = candidate_segment[70:].index, c = 'r', s = 1)
			axs[0 + cnt].scatter( y = -2,  x = candidate_segment[70:].index[0], c = 'r', s = 50, marker = '<')
			axs[0 + cnt].scatter( y = -2,  x = candidate_segment[70:].index[-1], c = 'r', s = 50, marker = '>')
		else:
			# plot non wear as scatter
			axs[0 + cnt].scatter( y = np.repeat(-2,len(candidate_segment)),  x = candidate_segment.index, c = 'r', s = 1)
			axs[0 + cnt].scatter( y = -2,  x = candidate_segment.index[0], c = 'r', s = 50, marker = '<')
			axs[0 + cnt].scatter( y = -2,  x = candidate_segment.index[-1], c = 'r', s = 50, marker = '>')
			
		"""
			Actiwave acceleration
		"""
		# plot acceleration Y
		axs[3 + cnt].plot(data['ACTIWAVE Y'].dropna(), label = 'Y', color = c[0])
		# plot acceleration X
		axs[3 + cnt].plot(data['ACTIWAVE X'].dropna(), label = 'X', color = c[1])
		# plot acceleration Z
		axs[3 + cnt].plot(data['ACTIWAVE Z'].dropna(), label = 'Z', color = c[2])
		# set the y axis limit
		axs[3 + cnt].set_ylim((-4,4))
		# set the title
		axs[3 + cnt].text(0.5 ,.9,'ActiWave Cardio accelerometer', horizontalalignment='center', transform = axs[3 + cnt].transAxes, fontdict = {'size' : 20})

		"""
			ESTIMATED HEART RATE DATA
		"""

		heart_rate_data = data['ESTIMATED HR'].dropna()
		if cnt == 2:
			heart_rate_data[0:60*38] = heart_rate_data[0:60*38].replace(0, np.nan)
			heart_rate_data = heart_rate_data.interpolate()
		# plot estimated heart rate
		axs[6 + cnt].plot(heart_rate_data, label='ESTIMATED HR', c = c[3])
		# set the y axis limit
		axs[6 + cnt].set_ylim((0,150))
		# set the title
		axs[6 + cnt].text(0.5 ,.9,'ActiWave Cardio Heart Rate', horizontalalignment='center', transform = axs[6 + cnt].transAxes, fontdict = {'size' : 20})

		# increment counter
		cnt += 1


	"""
		STYLING THE PLOT
	"""
	for i, ax in enumerate(axs):
		# no grid lines
		ax.grid(False)
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

		# only x axis on bottom plots
		if i not in [6, 7, 8]:
			ax.get_xaxis().set_visible(False)
		
		# only y axis on left plots
		if i not in [0, 3, 6]:
			ax.get_yaxis().set_visible(False)

	# titles on top of top plots
	axs[0].set_title('A', fontdict = {'size' : 20})
	axs[1].set_title('B', fontdict = {'size' : 20})
	axs[2].set_title('C', fontdict = {'size' : 20})

	# define y axis labels
	axs[0].set_ylabel('acceleration (g)', fontsize = 20)
	axs[3].set_ylabel('acceleration (g)', fontsize = 20)
	axs[6].set_ylabel('beats per minute', fontsize = 20)

	# candidate non wear episode label below arrows
	axs[0].text(0.53 ,.15,'candidate non-wear episode', horizontalalignment='center', transform = axs[0].transAxes, fontdict = {'size' : 16})
	axs[1].text(0.67 ,.15,'candidate non-wear episode', horizontalalignment='center', transform = axs[1].transAxes, fontdict = {'size' : 16})
	axs[2].text(0.78 ,.15,'candidate non-wear episode', horizontalalignment='center', transform = axs[2].transAxes, fontdict = {'size' : 16})

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, plot_name)
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()



"""
	FINAL PAPER PLOTS
"""
def plot_nw_distribution(data, plot_name = 'distribution-of-nw-times.pdf', plot_folder = os.path.join('plots', 'paper')):
	"""
	Plot histogram of non-wear episodes in minutes

	Parameters
	---------
	data : list
		list of non-wear episodes in minutes
	plot_name : string
		name of the plot with extension
	plot_folder : os.path
		location to save plot to
	"""

	fig, axs = plt.subplots(1, 2, figsize = (10,5))
	axs = axs.ravel()

	# first histogram
	bins_first = [1] + list(range(5,61,5))
	axs[0].hist(data, bins = bins_first, color = '#43a2ca')
	axs[0].set_xticks(bins_first)

	# second histogram
	bins_second = range(60,601,60)
	axs[1].hist(data, bins = bins_second, color = '#43a2ca')
	axs[1].set_xticks(bins_second)

	# adjust the axes
	for ax in axs:
		ax.set_xlabel('Non-wear time episode (mins)')
		ax.set_ylabel('Frequency')
		ax.grid(False)
		ax.tick_params(axis='both', left = True, bottom = True, which='both')

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, plot_name)
	# save the figure
	fig.savefig(save_location, dpi=150)
	# close the figure environemtn
	plt.close()


def plot_classification_results(data, data_filtered, plot_name = 'classification_performance.pdf', plot_folder = os.path.join('plots', 'paper')):
	"""
	Plot classification performance of the four non wear algoritms with their default parameters for two datasets
		1 = all the dat
		2 = a filtered dataset that contains only data between 07:00 - 23:00

	Parameters
	----------
	data : df.DataFrame
		pandas dataframe with performance data per non wear method
	data_filtered : df.DataFrame
		pandas dataframe with filtered data per non-wear method
	plot_name : string
		name of the plot with extension
	plot_folder : os.path
		location to save plot to
	"""

	# define which classification metrics to plot
	plot_metrics = ['accuracy', 'precision', 'recall', 'f1']

	# setting up the plot environment
	fig, axs = plt.subplots(1, len(plot_metrics), figsize=(14, 3), sharey = True)
	axs = axs.ravel()

	# create x ticks
	x = np.arange(len(data.index))
	# width of the bar
	width = .4

	# plot metrics
	for i, metric in enumerate(plot_metrics):

		# plot performance of all data
		bars = axs[i].bar(x, data[metric], width = width, label = 'all data', align = 'center', alpha = .8, color = '#d8b365' )
		# plot performance of data without nights
		bars += axs[i].bar(x + width, data_filtered[metric], width = width, label = '07:00 - 23:00', align = 'center', alpha = .8, color = '#5ab4ac')

		# plot bar height on top of bar
		for rect in bars:
			height = rect.get_height()
			axs[i].text(rect.get_x() + rect.get_width() / 2.0, height, round(height, 2), ha='center', va='bottom')
	
		# adjust the plot
		axs[i].set_xticks(x + width/2)
		axs[i].set_xticklabels(data.index)
		axs[i].grid(True, 'major', ls = '--')
		axs[0].set_ylabel('score')
		axs[i].set_ylim(0,1.1)
		axs[1].legend(loc='best', prop={'size': 10})
		axs[i].set_title('{}'.format(metric))
	
	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, plot_name)
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()


def plot_classification_results_comparison(df, plot_name ='classification_performance_comparison.pdf', plot_folder = os.path.join('plots', 'paper')):
	"""
	Plot bar charts that show the classification performance for the non-wear algorithms

	Parameters
	----------
	df : pd.DataFrame
		hold classification data for each classification metric and non-wear algoritm. Note that the values in the cells are lists
		with [0] being the default parameter and [1] the optimized
	plot_name : string
		name of the plot
	plot_folder : os.path()
		location to store the plot
	"""

	# setting up the plot environment
	fig, axs = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
	# width of each bar
	width = .4

	for cnt, data_tuple in enumerate(df.iterrows()):

		# get column name
		column = data_tuple[0]
		# get data
		data = data_tuple[1]
		# create bars
		for item_cnt, results in enumerate(data.items()):
			
			# plot default performance values
			bars = axs[cnt].bar(item_cnt, results[1][0], width = width, align = 'center', alpha = .8, color = '#d8b365' )
			# plot optimized performance values
			bars += axs[cnt].bar(item_cnt + width, results[1][1], width = width, align = 'center', alpha = .8, color = '#5ab4ac')

			# plot bar height on top of bar
			for rect in bars:
				height = rect.get_height()
				axs[cnt].text(rect.get_x() + rect.get_width() / 2.0, height, round(height, 2), ha='center', va='bottom')
		
		# adjust the plot
		axs[cnt].set_xticks(np.arange(len(data)) + width/2)
		axs[cnt].set_xticklabels(data.index)
		axs[cnt].grid(True, 'major', ls = '--')
		axs[0].set_ylabel('score')
		axs[cnt].set_ylim(0,1.1)
		axs[2].legend(['default', 'optimized'], loc='upper left', prop={'size': 10})
		axs[cnt].set_title('{}'.format(column))

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, plot_name)
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()


def plot_classification_results_comparison_all(df, plot_name ='classification_performance_comparison_all.pdf', plot_folder = os.path.join('plots', 'paper')):
	"""
	Plot bar charts that show the classification performance in comparison to other metrics, this shows how optimizing for precision affects the accuracy, recall, and f1.

	Parameters
	----------
	df : pd.DataFrame
		hold classification data for each classification metric and non-wear algoritm. Note that the values in the cells are lists
		with [0] being the default parameter and [1] the optimized
	plot_name : string
		name of the plot
	plot_folder : os.path()
		location to store the plot
	"""

	# setting up the plot environment
	fig, axs = plt.subplots(4, 4, figsize=(14, 12), sharey=True)
	axs = axs.ravel()
	# width of each bar
	width = .4

	for cnt, data_tuple in enumerate(df.iterrows()):

		# get column name
		column = data_tuple[0]
		# get data
		data = data_tuple[1]
		# create x ticks array
		x = np.arange(4)
	
		# create bars
		for item_cnt, results in enumerate(data.items()):

			# dynamically create plot index
			ax_idx = cnt * 3 + item_cnt + cnt

			# unpack non_wear method
			metric = results[0]
			# unpack default value
			default_values = [results[1][0]['accuracy'], results[1][0]['precision'], results[1][0]['recall'], results[1][0]['f1']]
			# unpack classification values
			optimized_values = [results[1][1]['accuracy'], results[1][1]['precision'], results[1][1]['recall'], results[1][1]['f1']]

			# plot default performance values
			bars = axs[ax_idx].bar(x, default_values, width = width, align = 'center', alpha = .8, color = '#d8b365' )
			# plot optimized performance values
			bars += axs[ax_idx].bar(np.array(x) + width, optimized_values, width = width, align = 'center', alpha = .8, color = '#5ab4ac')

			# plot bar height on top of bar
			for rect in bars:
				height = rect.get_height()
				axs[ax_idx].text(rect.get_x() + rect.get_width() / 2.0, height, round(height, 2), ha='center', va='bottom')
		
			# adjust the plot
			axs[ax_idx].set_xticks(np.arange(len(data)) + width/2)
			axs[ax_idx].set_xticklabels(['accuracy', 'precision', 'recall', 'f1'])
			axs[ax_idx].grid(True, 'major', ls = '--')
			axs[cnt * 3].set_ylabel('score')
			axs[ax_idx].set_ylim(0,1.1)
			axs[cnt].legend(['default', 'optimized'], loc='upper right', prop={'size': 10})
			axs[ax_idx].set_title('{} - Optimized for {}'.format(column, metric))

	# crop white space
	fig.set_tight_layout(True)
	# create the plot folder if not exist already
	create_directory(plot_folder)
	# create the save location
	save_location = os.path.join(plot_folder, plot_name)
	# save the figure
	fig.savefig(save_location)
	# close the figure environemtn
	plt.close()