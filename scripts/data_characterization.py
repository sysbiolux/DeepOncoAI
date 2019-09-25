# -*- coding: utf-8 -*-
"""
Created on Sa Sept 21 2019

@author: sebastien.delandtsheer@uni.lu

"""

import numpy as np
import pandas as pd

def explore_shape(df):
	#TODO: return shape of the dataframe and percentage of missing data
	#TODO: display and save matrix of dataframe with missing data indicated

	return (nLines, nColumns, percentMissing)

# Reduce dataframe memory usage
def reduce_mem_usage(df):
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

def reformat_drugs(df):













































