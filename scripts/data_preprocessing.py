# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""
import numpy as np
import pandas as pd


def eliminate_sparse_data(df, colThreshold = 0.5, lineThreshold = 0.5):
	"""drops the columns and lines that do not satisfy the thresholds in term of
	presence of data"""
	df_orig = df.copy()
	colFil = 1 - df.isna().mean()
	rowFil = 
	
		if df[col].isna().mean() < colThreshold:
			df_orig.drop([col])
	for row in df.r
	#TODO: drop columns and samples based on the thresholds
	#TODO: print the fraction of data that has been removed
	return df

def reformat_drugs(df):
	"""reshapes a pandas dataframe from 'one line per datapoint' to a more convenient
	'one line per sample' format, meaning the response of a given cell line to different drugs
	will be placed on the same line in different columns."""
	drugNames = df['Compound'].unique()
	df['Compound'].value_counts()
	# concatenate the drug info with one line per cell line
	Merged = pd.DataFrame()

	for thisDrug in drugNames:
		df_spec = df.loc[df['Compound'] == thisDrug]
		df_spec_clean = df_spec.drop(columns =['Primary Cell Line Name', 'Compound', 'Target', 'Doses (uM)', 'Activity Data (median)', 'Activity SD', 'Num Data', 'FitType'])
		df_spec_clean.columns=['CCLE Cell Line Name', thisDrug+'_EC50', thisDrug+'_IC50', thisDrug+'_Amax', thisDrug+'_ActArea']

		if Merged.empty:
			Merged = df_spec_clean.copy()
		else:
			Merged = pd.merge(Merged, df_spec_clean, how='left', on='CCLE Cell Line Name', sort=False, suffixes=('_x', '_y'), copy=True)
	merged_df = Merged.set_index('CCLE Cell Line Name')

	return merged_df


def impute_missing_data(df, method = 'average'):
	"""imputes computed values for missing data according to the specified method"""
	##TODO: 
	
	return df

def remove_outliers(df, method = 'normal', param = 0.05):
	"""removes outliers from each column independently according to different methods"""
	#TODO: 
	
	return df









