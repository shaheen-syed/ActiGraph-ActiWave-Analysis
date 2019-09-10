# -*- coding: utf-8 -*-

"""
	IMPORT PACKAGES
"""
import logging
import numpy as np
from scipy.stats import ttest_ind, ttest_ind_from_stats, levene


def calculate_euclidian_distance(t1, t2):
	"""
	Calculate the euclidian distance between two 1D vectors

	Parameters
	----------
	t1 : np.array(samples, 1)
		numpy vector data 1
	t2 : np.array(samples, 1)
		numpy vector data 2

	Returns
	----------
	euclidian distance : float
		numpy vector with euclidian distance
	"""

	return np.sqrt(np.sum((t1-t2)**2)) 


def calculate_lower_bound_Keogh(s1, s2, r):
	"""
	This function calculates a lower bound (LB) on the Dynamic Time Warp (DTW) distance between two time series.
	
	Parameters
	----------
	s1 : np.array(samples, 1)
		numpy vector data 1
	s2 : np.array(samples, 1)
		numpy vector data 2
	r : int
		distance

	Returns
	-------
	lb_keogh : float
		lower bound DTW
	"""
	
	LB_sum = 0

	for ind,i in enumerate(s1):

		lower_bound = np.min(s2[(ind - r if ind - r >=0 else 0):(ind + r)])
		upper_bound = np.max(s2[(ind - r if ind - r >=0 else 0):(ind + r)])

		if i > upper_bound:
			LB_sum = LB_sum + (i - upper_bound) ** 2
		elif i < lower_bound:
			LB_sum = LB_sum + (i - lower_bound) ** 2

	return np.sqrt(LB_sum)

def calculate_levene_test(sample1, sample2):
	"""
	Perform Levene test for equal variances.
	The Levene test tests the null hypothesis that all input samples are from populations with equal variances. Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.
	
	Parameters
	----------
	sample1: array with sample values
		The sample data, possibly with different lengths

	sample2: array with sample values
		The sample data, possibly with different lengths


	Returns
	-------
	W : float
		The test statistic.
	p-value : float
		The p-value for the test.
	"""

	# perform the test
	w, p = levene(sample1, sample2)

	# verbose
	logging.info("Levene's test statistics w: {} p: {}".format(w, p))

	# return the values
	return w, p


def calculate_ind_t_test(sample1, sample2, equal_var = True):
	"""
	Calculate the T-test for the means of two independent samples of scores.

	This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.

	Parameters
	----------
	sample1: array with sample values
			The sample data, possibly with different lengths

	sample2: array with sample values
			The sample data, possibly with different lengths


	Returns
	-------
	t : float or array
		The calculated t-statistic.

	p : float or array
		The two-tailed p-value.

	"""

	t, p = ttest_ind(sample1, sample2, equal_var = equal_var)

	# convert to scalar
	t, p = np.asscalar(t), np.asscalar(p)

	# verbose
	logging.info("Independent T-test statistics t: {} p: {}".format(t, p))

	# return the values
	return t, p


def get_significance_asterisk(p):
	"""
	Return a significance asterisk label for print purposes.
	For instance, we encode a p < 0.001 as ***

	Parametes
	----------
	p : float
		calculated p-value


	Returns
	------
	asterisk : string
		encoding for p value
	"""

	if p < 0.001:
		return r'$ \ddag$'
	elif p < 0.01:
		return r'$ \dag$'
	elif p < 0.05:
		return r'$ *$'
	else:
		return ''

def signal_to_noise(a, axis=0, ddof=0):

	a = np.asanyarray(a)
	
	m = a.mean(axis)
	
	sd = a.std(axis=axis, ddof=ddof)
	
	return np.where(sd == 0, 0, m/sd)
