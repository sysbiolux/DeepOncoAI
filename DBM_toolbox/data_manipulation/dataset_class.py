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
				resulting_dataset = individual_filter.apply(resulting_dataset)
		return resulting_dataset

	def to_pandas(self, omic=None, database=None):
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

	def merge_with(self, other_datasets):
		if isinstance(other_datasets, list):
			for single_dataset in other_datasets:
				self = self.merge_two_datasets(single_dataset)
		else:
			# TODO: check if dataset class, else throw error?
			self = self.merge_two_datasets(other_datasets)
		return self
	
	def merge_two_datasets(self, other_dataset):
		dataframe = self.dataframe
		other_dataframe = other_dataset.dataframe
		
		merged_dataframe = pd.merge(dataframe, other_dataframe, left_index = True, right_index = True)
		merged_omic = pd.concat([self.omic, other_dataset.omic])
		merged_database = pd.concat([self.database, other_dataset.database])
		
		merged_dataset = Dataset(dataframe=merged_dataframe, omic=merged_omic, database=merged_database)
		
		return merged_dataset
	
	### this is now done in the config file
# 	def get_features(self, feature_type, target_omic=None, target_database=None, options=None):
# 		n_components = None
# 		label = None
# 		if options is not None:
# 			n_components = options[0]
# 			label = options[1]
# 		dataframe = self.dataframe
# 		subset_omic = self.omic
# 		subset_database = self.database
# 		result_omic = 'overall'
# 		result_database = 'overall'
# 		subset = self
# 		if target_omic is not None:
# 			result_omic = 'engineered_' + target_omic
# 			subset = Dataset(dataframe=dataframe[subset_omic[subset_omic == target_omic].index], 
# 					omic = target_omic, database = subset_database[subset_omic == target_omic].index)
# 		if target_database is not None:
# 			result_database = 'source_' + target_database
# 			subset = Dataset(dataframe=dataframe[subset_database[subset_database == target_database].index], 
# 					omic = subset_omic[subset_database == target_database].index, database = target_database)
# 		dataframe = subset.dataframe
# 		result_omic = result_omic + '_' + feature_type
# 		
# 		if feature_type == 'PC':
# 			feature_df = components.get_PCs(dataframe, n_components=n_components, label=label)
# 		if feature_type == 'IC':
# 			feature_df = components.get_ICs(dataframe, n_components=n_components, label=label)
# 		if feature_type == 'RPC':
# 			feature_df = components.get_RPCs(dataframe, n_components=n_components, label=label)
# 		if feature_type == 'TSNE':
# 			feature_df = components.get_TSNEs(dataframe, n_components=n_components, label=label)
# 		if feature_type == 'POLY':
# 			feature_df = combinations.get_polynomials(dataframe, degree=n_components)
# 		if feature_type == 'OR':
# 			feature_df = combinations.get_boolean_or(dataframe)
# 		if feature_type == 'TYPE':
# 			feature_df = preprocessing.get_tumor_type(dataframe)
# 		
# 		return Dataset(dataframe=feature_df, omic=result_omic, database=result_database)

	def impute(self, method = 'average'):
		# TODO: make omic- and database-specific
		return Dataset(dataframe = preprocessing.impute_missing_data(self.dataframe, method=method), omic=self.omic, database=self.database)
	
	def normalize(self):
		# TODO: make omic- and database-specific
		return Dataset(dataframe = preprocessing.rescale_data(self.dataframe), omic=self.omic, database=self.database)

	def quantize(self, target_omic, quantiles=None):
		omic = self.omic
		database = self.database
		df = self.to_pandas()
		if quantiles is None:
			quantiles = [0.333, 0.667]
		
		q_df = df.copy()
		for this_feature in omic[omic == target_omic].index:
			q = np.quantile(df[this_feature], quantiles)
			q_df[this_feature] = 0.5
			q_df[this_feature].mask(df[this_feature] < q[0], 0, inplace=True)
			q_df[this_feature].mask(df[this_feature] >= q[1], 1, inplace=True)
				
		return Dataset(dataframe=q_df, omic=omic, database=database)




















# Visualization
# Comparison between multiple datasets

