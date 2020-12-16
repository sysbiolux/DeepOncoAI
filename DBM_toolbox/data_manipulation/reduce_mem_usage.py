# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:49:42 2020

@author: sebde
"""
import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def reduce_mem_usage(df):
	"""reduces memory usage for large pandas dataframes by changing datatypes per column into the ones
	that need the least number of bytes (int8 if possible, otherwise int16 etc...)"""

	df_orig = df.copy()
	start_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage is {:.2f} MB'.format(start_mem))
	
	for col in df.columns:
		if is_datetime(df[col]) or is_categorical_dtype(df[col]):
			continue
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
	print('Memory usage after optimization is {:.2f} MB'.format(end_mem))
	
	df_test = pd.DataFrame()
	
	print('checking consistency...')
	
	for col in df:
		col_type = df[col].dtype
#		print(col_type) for debugging
		if col_type != object:
			df_test[col] = df_orig[col] - df[col]
	
	#Mean, max and min for all columns should be 0
	mean_test = df_test.describe().loc['mean'].mean()
	max_test = df_test.describe().loc['max'].max()
	min_test = df_test.describe().loc['min'].min()
	
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	print('Min, Max and Mean of pre/post differences: {:.2f}, {:.2f}, {:.2f}'.format(min_test, max_test, mean_test))
	
	return df