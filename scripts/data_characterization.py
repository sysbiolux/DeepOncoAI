# -*- coding: utf-8 -*-
"""
Created on Sa Sept 21 2019

@author: sebastien.delandtsheer@uni.lu

"""

import numpy as np
import pandas as pd
import warnings
import scipy.stats as st
import statsmodels as sm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def explore_shape(df):
	"""accepts a pandas dataframe and returns its dimensions as well as a
	graph of the presence of data."""
	#TODO: return shape of the dataframe and percentage of missing data
	#TODO: display and save matrix of dataframe with missing data indicated

	return (nLines, nColumns, percentMissing)

# Reduce dataframe memory usage
def reduce_mem_usage(df):
	"""reduces memory usage for large pandas dataframes by changing datatypes per column into the ones
	that need the least number of bytes (int8 if possible, otherwise int16 etc...)"""
	### iterate through all the columns of a dataframe and modify the data type to reduce memory usage.

	df_orig = df
	start_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

	for col in df.columns:
		col_type = df[col].dtype

		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('category')

	end_mem = df.memory_usage().sum() / 1024**2

	df_test = pd.DataFrame()

	for col in df:
		df_test[col] = df_orig[col] - df[col]

	#Mean, max and min for all columns should be 0
	mean_test = df_test.describe().loc['mean']
	max_test = df_test.describe().loc['max']
	min_test = df_test.describe().loc['min']

	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	print('Min, Max and Mean of pre/post differences: {:.2f}, {:.2f}, {:.2f}'.format(min_test, max_test, mean_test))

	return df

# Create models from data
def best_fit_distribution(df, bins=200, ax=None):
	"""Model data by finding best fit distribution to data"""
	# Get histogram of original data
	y, x = np.histogram(df, bins=bins, density=True)
	x = (x + np.roll(x, -1))[:-1] / 2.0

	# Distributions to check
	DISTRIBUTIONS = [st.alpha,st.beta,st.chi2,st.expon,st.exponnorm,st.gamma,st.logistic,st.loggamma,st.lognorm,st.norm,st.powerlaw,st.uniform]

	# Best holders
	best_distribution = st.norm
	best_params = (0.0, 1.0)
	best_sse = np.inf
	best_pval = 0

	# Estimate distribution parameters from data
	for distribution in DISTRIBUTIONS:

		print(distribution.name)

		# Try to fit the distribution
		try:
			# Ignore warnings from data that can't be fit
			with warnings.catch_warnings():
				#warnings.filterwarnings('ignore')

				# fit dist to data
				params = distribution.fit(df)

				# Separate parts of parameters
				arg = params[:-2]
				loc = params[-2]
				scale = params[-1]

				# Calculate fitted PDF and error with fit in distribution
				pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
				sse = np.sum(np.power(y - pdf, 2.0))
				ks, pval = st.kstest(df.values, distribution.cdf)
				print(ks, pval)

				# if axis pass in add to plot
				try:
					if ax:
						pd.Series(pdf, x).plot(ax=ax)
					end
				except Exception:
					pass

				# identify if this distribution is better
				if pval > best_pval:
					best_distribution = distribution
					best_params = params
					best_sse = sse
					best_pval = pval

		except Exception:
			pass

	return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
	"""Generate distributions's Probability Distribution Function """

	# Separate parts of parameters
	arg = params[:-2]
	loc = params[-2]
	scale = params[-1]

	# Get sane start and end points of distribution
	start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
	end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

	# Build PDF and turn into pandas Series
	x = np.linspace(start, end, size)
	y = dist.pdf(x, loc=loc, scale=scale, *arg)
	pdf = pd.Series(y, x)

	return pdf







def show_me_the_data(df):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from data_characterization import best_fit_distribution, make_pdf

	sns.violinplot(data=df)

	for col in df.columns:
		df2 = df[col]
		bestDistrib, bestParams = best_fit_distribution(df2)
		pdf = make_pdf(bestDistrib, bestParams)
		f, axes = plt.subplots(1, 2)
		sns.distplot(df2, kde=True, rug=True, ax=axes[0], label='data')
		sns.distplot(pdf, hist=False, kde=True, ax=axes[0], label=bestDistrib.name)
		plt.legend()

























































