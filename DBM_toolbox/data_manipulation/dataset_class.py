import logging

import pandas as pd
import numpy as np
# from DBM_toolbox.feature_engineering.predictors import combinations, components
from DBM_toolbox.data_manipulation import preprocessing


class Dataset:
	def __init__(self, dataframe, omic, database):
		self.dataframe = dataframe
		n_rows, n_cols = dataframe.shape
		if isinstance(omic, str):
			self.omic = pd.Series(data=[omic for x in range(n_cols)], index=dataframe.columns)
		elif len(omic) == n_cols:
			self.omic = omic
		else:
			raise ValueError('Omic should be either a string or a Series with the same lengths as the number of features')
		if isinstance(database, str):
			self.database = pd.Series(data=[database for x in range(n_cols)], index=dataframe.columns)
		elif len(database) == n_cols:
			self.database = database
		else:
			raise ValueError('Database should be either a string or a Series with the same lengths as the number of features')
		
	def __str__(self):
		return f'Dataset with omic {self.omic}, from database {self.database}'
	
	def apply_filters(self, filters=None):
		resulting_dataset = self
		if filters:
			for individual_filter in filters:
# 				logging.info(f"{individual_filter}")
				resulting_dataset = individual_filter.apply(resulting_dataset)
		return resulting_dataset

	def to_pandas(self, omic=None, database=None): #TODO: possibility to use lists of omics and databases?
		"""
		returns the Pandas Dataframe of the dataset for columns matching BOTH the omic and database
		
		"""
	
		resulting_dataframe = self.dataframe
		resulting_database = self.database
		if omic is not None:
			if omic not in list(self.omic):
				raise ValueError(f'Omics type {omic} not present')
			resulting_dataframe = resulting_dataframe.loc[:, self.omic == omic]
			resulting_database = resulting_database.loc[self.omic == omic]
		if database is not None:
			if database not in list(self.database):
				raise ValueError(f'Database {database} not present')
			resulting_dataframe = resulting_dataframe.loc[:, resulting_database == database]
		return resulting_dataframe
	
	def extract(self, omics_list=[], databases_list=[]):
		"""
		retuns the parts of the dataset matching EITHER ONE of the the elements of the omics_list and databases_list
		 
		"""
		resulting_dataframe = self.dataframe
		resulting_database = self.database
		resulting_omic = self.omic
		if (omics_list is not None) or (databases_list is not None):
			to_extract = resulting_omic.isin(omics_list) | resulting_database.isin(databases_list)
			resulting_dataframe = resulting_dataframe.loc[:,to_extract==True]
			resulting_database = resulting_database.loc[to_extract==True]
			resulting_omic = resulting_omic.loc[to_extract==True]
		return Dataset(dataframe=resulting_dataframe, omic=resulting_omic, database=resulting_database)

	def merge_with(self, other_datasets):
		if isinstance(other_datasets, list):
			for single_dataset in other_datasets:
				self = self.merge_two_datasets(single_dataset)
		else:
			# TODO: check if dataset class, else throw error?
			self = self.merge_two_datasets(other_datasets)
		return self
	
	def merge_two_datasets(self, other_dataset):
		print(self, other_dataset)
		dataframe = self.dataframe
		other_dataframe = other_dataset.dataframe
		
		merged_dataframe = pd.merge(dataframe, other_dataframe, left_index = True, right_index = True)
		merged_omic = pd.concat([self.omic, other_dataset.omic])
		merged_database = pd.concat([self.database, other_dataset.database])
		
		merged_dataset = Dataset(dataframe=merged_dataframe, omic=merged_omic, database=merged_database)
		
		return merged_dataset
	
	def impute(self, method = 'average'):
		return Dataset(dataframe = preprocessing.impute_missing_data(self.dataframe, method=method), omic=self.omic, database=self.database)
	
	def normalize(self):
		return Dataset(dataframe = preprocessing.rescale_data(self.dataframe), omic=self.omic, database=self.database)

	def quantize(self, target_omic, quantiles=None):
		"""
		will ternarize the columns with the corresponding prefix (ex: 'DRUGS')
		quantiles should be a list of two values in [0, 1]
		"""
		omic = self.omic
		database = self.database
		dataframe = self.to_pandas()
		if quantiles is None:
			quantiles = [0.333, 0.667]
		
		quantized_dataframe = dataframe.copy()
		for target in omic[omic.str.startswith(target_omic)].index:
			q = np.quantile(dataframe[target].dropna(), quantiles)
			quantized_dataframe[target] = 0.5 #start assigning 'intermediate' to all samples
			quantized_dataframe[target].mask(dataframe[target] < q[0], 0, inplace=True) #samples below the first quantile get 0
			quantized_dataframe[target].mask(dataframe[target] >= q[1], 1, inplace=True) #samples above get 1
# 			# uncomment to get random values
# 			q_df[target] = np.random.randint(low=0, high=2, size=len(q_df))
				
		return Dataset(dataframe=quantized_dataframe, omic=omic, database=database)

	def to_binary(self, target:str):
		omic=self.omic
		database = self.database
		dataframe = self.to_pandas()
		
		m = min(dataframe[target])
		n = max(dataframe[target])
		
		dataframe = dataframe[dataframe[target].isin([m, n])]
		
		return Dataset(dataframe=dataframe, omic=omic, database=database)
		
	def split(self, train_index, test_index):
		omic = self.omic
		database = self.database
		dataframe = self.to_pandas()
		train_dataset = Dataset(dataframe=dataframe[train_index], omic=omic, database=database)
		test_dataset = Dataset(dataframe=dataframe[test_index], omic=omic, database=database)
		
		return train_dataset, test_dataset