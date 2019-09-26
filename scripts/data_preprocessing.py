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
	#TODO: drop columns and samples based on the thresholds
	#TODO: print the fraction of data that has been removed
	return df

def reformat_drugs(df):
	"""reshapes a pandas dataframe from 'one line per datapoint' to a more convenient
	'one line per sample' format, meaning the response of a given cell line to different drugs
	will be placed on the same line in different columns."""
	drugNames = dfDrug_opt['Compound'].unique()
	dfDrug_opt['Compound'].value_counts()





	return merged_df
