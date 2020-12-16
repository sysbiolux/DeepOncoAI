# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:13:55 2020

@author: sebde
"""
import warnings


def impute_missing_data(df, method = 'average'):
	"""imputes computed values for missing data according to the specified method"""
	##TODO: implement other methods of imputation
	if method == 'average':
		df = df.fillna(df.mean())
	elif method == 'median':
		raise ValueError('Function not configured for this use')
		df = df
	elif method == 'neighbor':
		raise ValueError('Function not configured for this use')
		df = df

	return df