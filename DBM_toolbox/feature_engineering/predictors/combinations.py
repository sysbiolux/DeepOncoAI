# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:56:32 2020

@author: sebde
"""

from DBM_toolbox.data_manipulation import dataset_class

import pandas as pd
from datetime import datetime

from sklearn.preprocessing import PolynomialFeatures

def make_dataset(dataframe, omic=None, database=None):
	dataset = dataset_class.Dataset(dataframe=dataframe, omic=omic, database=database)
	return dataset

def get_polynomials(df, degree = None, label=None):
	"""adds polynomial features to the dataset"""
	if degree == None:
		degree = 2
	if label == None:
		label = (datetime.now()).strftime("%Y%m%d%H%M%S")
	poly = PolynomialFeatures(degree)
	df_transformed = poly.fit_transform(df)
	df_polynomial = pd.DataFrame(data = df_transformed, index = df.index, columns = poly.get_feature_names())
	df.columns += '_' + label
	return make_dataset(df_polynomial, omic='poly', database='ENGINEEERED')

def get_boolean_or(df, label=None):
	"""adds boolean OR combinations of the features to the dataset
	OR (1-(1-a)*(1-b))"""
	if label == None:
		label = (datetime.now()).strftime("%Y%m%d%H%M%S")
	df_boolean = pd.DataFrame()
	count = 0
	for col1 in range(len(df.columns)-1):
		for col2 in range(col1, len(df.columns)):
			count += 1
			A = df.iloc[:,col1]
			B = df.iloc[:,col2]
			or_value = 1 - ((1 - A) * (1 - B))
			df_boolean['BoolOR' + str(count) + label] = or_value
	
	return make_dataset(df_boolean, omic='boolean', database='ENGINEEERED')
