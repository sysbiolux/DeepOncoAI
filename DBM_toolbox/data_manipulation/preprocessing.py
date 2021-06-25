# -*- coding: utf-8 -*-
'''

'''

import pandas as pd
import numpy as np
from DBM_toolbox.data_manipulation import dataset_class
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from sklearn.impute import KNNImputer

def reformat_drugs(dataset):
	'''reshapes a CCLE pandas dataframe from 'one line per datapoint' to a more convenient
	'one line per sample' format, meaning the response of a given cell line to different drugs
	will be placed on the same line in different columns.'''
	
	dataframe = dataset.dataframe
	database = dataset.database
	omic = dataset.omic
	if all(x == database[0] for x in database) and all(x.split('_')[0] == 'DRUGS' for x in omic):
		if database[0] == 'CCLE':
			drugNames = dataframe['Compound'].unique()
			dataframe['Compound'].value_counts()
			# concatenate the drug info with one line per cell line
			merged = pd.DataFrame()
		
			for thisDrug in drugNames:
				dataframe_spec = dataframe.loc[dataframe['Compound'] == thisDrug]
				dataframe_spec_clean = dataframe_spec.drop(columns =['Primary Cell Line Name', 'Compound', 'Target', 'Doses (uM)', 'Activity Data (median)', 'Activity SD', 'Num Data', 'FitType'])
				dataframe_spec_clean.columns=['CCLE Cell Line Name', thisDrug+'_EC50', thisDrug+'_IC50', thisDrug+'_Amax', thisDrug+'_ActArea']
		
				if merged.empty:
					merged = dataframe_spec_clean.copy()
				else:
					merged = pd.merge(merged, dataframe_spec_clean, how='left', on='CCLE Cell Line Name', sort=False, suffixes=('_x', '_y'), copy=True)
			merged_dataframe = merged.set_index('CCLE Cell Line Name')
			n_rows, n_cols = merged_dataframe.shape
			omic = pd.Series(data=['DRUGS' for x in range(n_cols)], index=merged_dataframe.columns)
			database = pd.Series(data=['CCLE' for x in range(n_cols)], index=merged_dataframe.columns)
		
		elif database[0] == 'GDSC':
			pass
	
	return dataset_class.Dataset(merged_dataframe, omic=omic, database=database)
	
	
def preprocess_data(dataset, flag: str=None):
	omic = dataset.omic
	database = dataset.database
	if all(x == database[0] for x in database) and all(x == omic[0] for x in omic):
		if database[0] == 'CCLE':
			if omic[0] == 'RPPA':
				dataset = preprocess_ccle_rppa(dataset, flag=flag)
			elif omic[0] == 'RNA':
				dataset = preprocess_ccle_rna(dataset, flag=flag)
			elif omic[0] == 'RNA_FILTERED':
				dataset = preprocess_ccle_rna_filtered(dataset, flag=flag)
			elif omic[0] == 'MIRNA':
				dataset = preprocess_ccle_mirna(dataset, flag=flag)
			elif omic[0] == 'META':
				dataset = preprocess_ccle_meta(dataset, flag=flag)
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
		elif database[0] == 'OWN':
			if omic[0] == 'PATHWAYS':
				dataset = preprocess_features_pathway(dataset, flag=flag)
			if omic[0] == 'TOPOLOGY':
				dataset = preprocess_features_topology(dataset, flag=flag)
	return dataset


def preprocess_ccle_rppa(dataset, flag: str=None):
	if flag == None:
		df = dataset.dataframe
		df = df.set_index('Unnamed: 0')
		df = rescale_data(df)
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='RPPA', database='CCLE')

def preprocess_ccle_rna(dataset, flag: str=None):
	df = dataset.dataframe
	if flag == None:
		df['GeneTrans'] = df['Description'] + '_' + df['Name']
		df = df.set_index(['GeneTrans'])
		df = df.drop(['Description', 'Name'], axis=1)
		df = df.transpose()
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='RNA', database='CCLE')

def preprocess_ccle_rna_filtered(dataset, flag: str=None):
	if flag == None:
		df = dataset.dataframe
		df = df.set_index('Unnamed: 0')
		return dataset_class.Dataset(df, omic='RNA-FILTERED', database='CCLE')

def preprocess_ccle_mirna(dataset, flag: str=None):
	df = dataset.dataframe
	if flag == None:
		df['GeneTrans'] = df['Description'] + '_' + df['Name']
		df = df.set_index(['GeneTrans'])
		df = df.drop(['Description', 'Name'], axis=1)
		df = df.transpose()
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='MIRNA', database='CCLE')

def preprocess_ccle_meta(dataset, flag: str=None):
	df = dataset.dataframe
	if flag == None:
		df = df.drop('DepMap_ID', axis=1).set_index(['CCLE_ID'])
	
	return dataset_class.Dataset(df, omic='META', database='CCLE')

def preprocess_ccle_dna(dataset, flag: str=None):
	## TODO: preprocessing steps here
	pass

def preprocess_gdsc_rna(dataset, flag: str=None):
	df = dataset.dataframe
	if flag == None:
# 		df['GeneTrans'] = df['Description'] + '_' + df['Name']
# 		df = df.set_index(['GeneTrans'])
# 		df = df.drop(['Description', 'Name'], axis=1)
# 		df = df.transpose()
		df = np.log2(df + 1)
		
	return dataset_class.Dataset(df, omic='RNA', database='GDSC')

def preprocess_gdsc_mirna(dataset, flag: str=None):
	## TODO: preprocessing steps here
	pass

def preprocess_gdsc_dna(dataset, flag: str=None):
	## TODO: preprocessing steps here
	pass

def preprocess_features_pathway(dataset, flag: str=None):
	df = dataset.dataframe
	if flag == None:
		df = df.set_index(['Cell_line'])
	
	return dataset_class.Dataset(df, omic='PATHWAYS', database='OWN')
	

def preprocess_features_topology(dataset, flag: str=None):
	# @Apurva
	df = dataset.dataframe
	df = df.drop('Unnamed: 0', axis=1).set_index(['Gene']).transpose()
	df.index = [idx[6:-11] for idx in df.index]
	df = df.add_suffix('_topo')
# 	df = impute_missing_data(df, method='zeros')
	# additional steps if necessary
	
	return dataset_class.Dataset(df, omic='TOPOLOGY', database='OWN')

def rescale_data(dataframe):
	'''Normalization by mapping to the [0 1] interval (each feature independently)
	this is the same as maxScaler? should we leave it?'''
	return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

def impute_missing_data(dataframe, method: str='average'):
	'''imputes computed values for missing data according to the specified method'''
	##TODO: implement other methods of imputation
	if method == 'average':
		dataframe = dataframe.fillna(dataframe.mean())
	elif method = 'null':
		dataframe = dataframe.fillna(0)
	elif method == 'median':
		raise ValueError('Function not configured for this use')
		dataframe = dataframe.fillna(dataframe.median())
	elif method == 'neighbor':
		raise ValueError('Function not configured for this use')
		imputer = KNNImputer()
		imputer.fit(dataframe)
		dataframe = imputer.transform(dataframe)
	elif method == 'zeros':
		dataframe = dataframe.fillna(value=0)
	return dataframe

def remove_constant_data(dataframe):
	'''removes the columns that are strictly constant'''
	dataframe = dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]
	return dataframe

def get_tumor_type(dataframe):
	tumors_list = ['PROSTATE', 'STOMACH', 'URINARY', 'NERVOUS', 'OVARY', 'HAEMATOPOIETIC',
	'KIDNEY', 'THYROID', 'SKIN', 'SOFT_TISSUE', 'SALIVARY', 'LUNG', 'BONE',
	'PLEURA', 'ENDOMETRIUM', 'BREAST', 'PANCREAS', 'AERODIGESTIVE', 'LARGE_INTESTINE',
	'GANGLIA', 'OESOPHAGUS', 'FIBROBLAST', 'CERVIX', 'LIVER', 'BILIARY', 
	'SMALL_INTESTINE']
	
	dataframe_tumors = pd.DataFrame(index=dataframe.index, columns=tumors_list)
	for this_tumor_type in tumors_list:
		for this_sample in dataframe.index:
			if this_tumor_type in this_sample:
				dataframe_tumors.loc[this_sample, this_tumor_type] = 1.0
			else:
				dataframe_tumors.loc[this_sample, this_tumor_type] = 0.0
	
	for col in dataframe_tumors.columns:
		dataframe_tumors[col] = pd.to_numeric(dataframe_tumors[col])
	
	return dataframe_tumors

def select_drug_metric(dataset, metric: str):
	omic = dataset.omic
	database = dataset.database
	dataframe = dataset.dataframe
	is_selected = dataframe.columns.str.contains(metric, regex=False)
	dataframe = dataframe.loc[:, is_selected]
	omic = omic.loc[is_selected]
	database = database.loc[is_selected]
	sparse_dataset = dataset_class.Dataset(dataframe, omic=omic, database=database)
	
	return sparse_dataset

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
#		print(col_type)
		if col_type != object:
			df_test[col] = df_orig[col] - df[col]
	
	#Mean, max and min for all columns should be 0
	mean_test = df_test.describe().loc['mean'].mean()
	max_test = df_test.describe().loc['max'].max()
	min_test = df_test.describe().loc['min'].min()
	
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	print('Min, Max and Mean of pre/post differences: {:.2f}, {:.2f}, {:.2f}'.format(min_test, max_test, mean_test))
	
	return df