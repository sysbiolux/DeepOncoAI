# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""
def transform_zscores(df):
	"""replaces the data in each column with the z-score for that column"""
	import numpy as np
	import pandas as pd
	from scipy.stats import zscore
	df.apply(zscore)
	return df


def get_drug_response(df, thresholdR, thresholdS):
	"""replaces the data with indication of sensitivity:
		-1 = Resistant
		0 = Intermediate
		1 = Sensitive
		values in df indicate response to drug (like Act Area)"""
	from sklearn.preprocessing import binarize as bn
	dfSens = bn(df, threshold = thresholdS) #Resistants will get 0, while Intermediates and Sensitives get 1
	dfNotR = bn(df, threshold = thresholdR) #Sensitives get 1 more, others get 0
	dfAll = dfSens + dfNotR -1
	return dfAll







