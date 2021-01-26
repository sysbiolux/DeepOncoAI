# -*- coding: utf-8 -*-
'''

'''

import pandas as pd
import numpy as np
from DBM_toolbox.data_manipulation import dataset_class


def reformat_drugs(dataset):
	df = dataset.dataframe
	database = dataset.database
	omic = dataset.omic
	if all(x == database[0] for x in database) and all(x == 'DRUGS' for x in omic):
		if database[0] == 'CCLE':
			
			'''reshapes a CCLE pandas dataframe from 'one line per datapoint' to a more convenient
			'one line per sample' format, meaning the response of a given cell line to different drugs
			will be placed on the same line in different columns.'''
			
			drugNames = df['Compound'].unique()
			df['Compound'].value_counts()
			# concatenate the drug info with one line per cell line
			merged = pd.DataFrame()
		
			for thisDrug in drugNames:
				df_spec = df.loc[df['Compound'] == thisDrug]
				df_spec_clean = df_spec.drop(columns =['Primary Cell Line Name', 'Compound', 'Target', 'Doses (uM)', 'Activity Data (median)', 'Activity SD', 'Num Data', 'FitType'])
				df_spec_clean.columns=['CCLE Cell Line Name', thisDrug+'_EC50', thisDrug+'_IC50', thisDrug+'_Amax', thisDrug+'_ActArea']
		
				if merged.empty:
					merged = df_spec_clean.copy()
				else:
					merged = pd.merge(merged, df_spec_clean, how='left', on='CCLE Cell Line Name', sort=False, suffixes=('_x', '_y'), copy=True)
			merged_df = merged.set_index('CCLE Cell Line Name')
			n_rows, n_cols = merged_df.shape
			omic = pd.Series(data=['DRUGS' for x in range(n_cols)], index=merged_df.columns)
			database = pd.Series(data=['CCLE' for x in range(n_cols)], index=merged_df.columns)
		
		elif database[0] == 'GDSC':
			pass
	
	return dataset_class.Dataset(merged_df, omic=omic, database=database)
	
	
def preprocess_data(dataset, flag=None):
	omic = dataset.omic
	database = dataset.database
	if all(x == database[0] for x in database) and all(x == omic[0] for x in omic):
		if database[0] == 'CCLE':
			if omic[0] == 'RPPA':
				dataset = preprocess_ccle_rppa(dataset, flag=flag)
			elif omic[0] == 'RNA':
				dataset = preprocess_ccle_rna(dataset, flag=flag)
			elif omic[0] == 'MIRNA':
				dataset = preprocess_ccle_mirna(dataset, flag=flag)
			elif omic[0] == 'DNA':
				dataset = preprocess_ccle_dna(dataset, flag=flag)
			else:
				pass
		elif database[0] == 'GDSC':
			if omic[0] == 'RNA':
				dataset = preprocess_gdsc_rna(dataset, flag=flag)
			if omic[0] == 'MIRNA':
				dataset = preprocess_gdsc_mirna(dataset, flag=flag)
			if omic[0] == 'DNA':
				dataset = preprocess_gdsc_dna(dataset, flag=flag)
	return dataset


def preprocess_ccle_rppa(dataset, flag=None):
	if flag == None:
		df = dataset.dataframe
		df = df.set_index('Unnamed: 0')
		df = rescale_data(df)
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='RPPA', database='CCLE')

def preprocess_ccle_rna(dataset, flag=None):
	df = dataset.dataframe
	if flag == None:
		df['GeneTrans'] = df['Description'] + '_' + df['Name']
		df = df.set_index(['GeneTrans'])
		df = df.drop(['Description', 'Name'], axis=1)
		df = df.transpose()
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='RNA', database='CCLE')

def preprocess_ccle_mirna(dataset, flag=None):
	df = dataset.dataframe
	if flag == None:
		df['GeneTrans'] = df['Description'] + '_' + df['Name']
		df = df.set_index(['GeneTrans'])
		df = df.drop(['Description', 'Name'], axis=1)
		df = df.transpose()
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='MIRNA', database='CCLE')

def preprocess_ccle_dna(dataset, flag=None):
	## TODO: preprocessing steps here
	pass

def preprocess_gdsc_rna(dataset, flag=None):
	df = dataset.dataframe
	if flag == None:
# 		df['GeneTrans'] = df['Description'] + '_' + df['Name']
# 		df = df.set_index(['GeneTrans'])
# 		df = df.drop(['Description', 'Name'], axis=1)
# 		df = df.transpose()
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='RNA', database='GDSC')

def preprocess_gdsc_mirna(dataset, flag=None):
	## TODO: preprocessing steps here
	pass

def preprocess_gdsc_dna(dataset, flag=None):
	## TODO: preprocessing steps here
	pass

def rescale_data(df):
	'''Normalization by mapping to the [0 1] interval (each feature independently)
	this is the same as maxScaler? should we leave it?'''
	return (df - df.min()) / (df.max() - df.min())

def impute_missing_data(df, method = 'average'):
	'''imputes computed values for missing data according to the specified method'''
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

def get_tumor_type(df):
	tumors_list = ['PROSTATE', 'STOMACH', 'URINARY', 'NERVOUS', 'OVARY', 'HAEMATOPOIETIC',
	'KIDNEY', 'THYROID', 'SKIN', 'SOFT_TISSUE', 'SALIVARY', 'LUNG', 'BONE',
	'PLEURA', 'ENDOMETRIUM', 'BREAST', 'PANCREAS', 'AERODIGESTIVE', 'LARGE_INTESTINE',
	'GANGLIA', 'OESOPHAGUS', 'FIBROBLAST', 'CERVIX', 'LIVER', 'BILIARY', 
	'SMALL_INTESTINE']
	
	df_tumors = pd.DataFrame(index=df.index, columns=tumors_list)
	for this_tumor_type in tumors_list:
		for this_sample in df.index:
			if this_tumor_type in this_sample:
				df_tumors.loc[this_sample, this_tumor_type] = 1.0
			else:
				df_tumors.loc[this_sample, this_tumor_type] = 0.0
	
	for col in df_tumors.columns:
		df_tumors[col] = pd.to_numeric(df_tumors[col])
	
	return df_tumors

def select_drug_metric(dataset, metric):
	omic = dataset.omic
	database = dataset.database
	df = dataset.dataframe
	is_selected = df.columns.str.contains(metric, regex=False)
	df = df.loc[:, is_selected]
	omic = omic.loc[is_selected]
	database = database.loc[is_selected]
	sparse_dataset = dataset_class.Dataset(df, omic=omic, database=database)
	
	return sparse_dataset

