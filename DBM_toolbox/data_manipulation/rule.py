import numpy as np
import pandas as pd
import random
import logging
from psutil import virtual_memory
from DBM_toolbox.data_manipulation.filter_class import KeepFeaturesFilter
import xgboost as xgb
# from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error

class Rule:
	def create_filter(self, dataset):
		pass

# TODO: these rules could also be database-specific
class HighestVarianceRule(Rule):
	def __init__(self, fraction, omic, database):
		# TODO: Add check on fraction
		self.fraction = fraction
		self.omic = omic
		self.database = database

	def create_filter(self, dataset):
		dataframe = dataset.to_pandas(omic=self.omic)
		variances = dataframe.var().sort_values(ascending=False)
		number_of_features_to_keep = int(round(len(variances) * self.fraction))
		features_to_keep = variances.iloc[:number_of_features_to_keep].index
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)


class ColumnDensityRule(Rule):
	def __init__(self, completeness_threshold:float, omic:str, database:str):
		if completeness_threshold < 0 or completeness_threshold > 1:
			raise ValueError('ColumnDensityRule completeness_threshold should be in [0, 1]')
		self.density_fraction = completeness_threshold
		self.omic = omic
		self.database = database

	def create_filter(self, dataset):
		dataframe = dataset.to_pandas(omic=self.omic)
		completeness = dataframe.isna().mean(axis = 0).sort_values(ascending=False)
		number_of_features_to_keep = int(round(len(completeness) * self.density_fraction))
		features_to_keep = completeness.iloc[:number_of_features_to_keep].index
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)

class CrossCorrelationRule(Rule):
	def __init__(self, correlation_threshold, omic, database):
		if correlation_threshold < 0 or correlation_threshold > 1:
			raise ValueError('CrossCorrelationRule correlation_threshold should be in [0, 1]')
		self.correlation_threshold = correlation_threshold
		self.omic = omic
		self.database = database
		try: 
			free_mem = virtual_memory().free
			self.chunk_size = 500*round((np.sqrt(free_mem/64))/1000)
		except:
			self.chunk_size = 2000
		
	def create_filter(self, dataset):
		
		def get_features(dataframe, corr_threshold):
			n_features_start = dataframe.shape[1]
			corr_array = np.abs(np.corrcoef(dataframe.values, rowvar=False)) #   corr_matrix.to_numpy()
			np.fill_diagonal(corr_array, 0)
			clean = (np.nanmax(corr_array) < self.correlation_threshold)
			while not clean:
				sum_correlations = np.sum(corr_array, axis=0)
				ind = np.unravel_index(np.argmax(corr_array), corr_array.shape)
				if sum_correlations[ind[0]] > sum_correlations[ind[1]]:
					to_remove = ind[0]
				else:
					to_remove = ind[1]
				feature_to_remove = dataframe.columns[to_remove]
				dataframe = dataframe.drop(feature_to_remove, axis=1)
				for axis in [0, 1]:
					corr_array = np.delete(corr_array, to_remove, axis=axis)
				clean = (np.nanmax(corr_array) < self.correlation_threshold)
			
			features_to_keep_in = dataframe.columns
			n_features_removed = n_features_start - len(features_to_keep_in)
			return features_to_keep_in
		
		def shuffle_columns(dataframe, seed:int=42):
			col_names = list(dataframe.columns)
			random.seed(seed)
			random.shuffle(col_names)
			dataframe = dataframe[col_names]

			return dataframe
		
		dataframe = dataset.extract(omics_list=[self.omic]).remove_constants().normalize().impute().to_pandas()
		
		if len(dataframe.columns) < self.chunk_size:
			features_to_keep = get_features(dataframe, self.correlation_threshold)
		else: # cannot compute pandas.corr() for large matrices
			keep_on_trying = True
			while keep_on_trying:
				dataframe = shuffle_columns(dataframe=dataframe)
				features_to_keep = []
				n_chunks = round(len(dataframe.columns)/self.chunk_size + 0.5)
				starts = [x * self.chunk_size for x in (range(n_chunks))]
				stops = [x + (self.chunk_size) for x in starts]
				stops[-1] = len(dataframe.columns)
				for count, col_start in enumerate(starts):
					col_stop = stops[count]
					mini_dataframe = dataframe.iloc[:, col_start:col_stop]
					features_to_keep.extend(get_features(mini_dataframe, self.correlation_threshold))
				
				new_dataframe = dataframe[features_to_keep]
				keep_on_trying = (new_dataframe.size != dataframe.size)
				dataframe = new_dataframe
		return KeepFeaturesFilter(features=dataframe.columns, omic=self.omic, database=self.database)
	
class FeatureImportanceRule(Rule):
	def __init__(self, fraction, omic, database):
		# TODO: Add check on fraction allowed values
		self.fraction = fraction
		self.omic = omic
		self.database = database
		
	def create_filter(self, dataset, target_dataframe):
		dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
		if len(target_dataframe.shape) == 1:
			target_dataframe = target_dataframe.to_frame()
# 		index = target_dataframe.index[target_df.apply(np.isnan)]  ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost
# 		to_drop = index.values.tolist()
# 		dataframe = dataframe.drop(to_drop)
# 		target_dataframe = target_dataframe.drop(to_drop)
		
		importances = pd.DataFrame()
		for this_target in target_dataframe.columns:
			model = xgb.XGBClassifier(max_depth=4, n_estimators=100, colsample_bytree = 0.5) ### deeper?
			model.fit(dataframe, target_dataframe) #use ravel() here?
			scores = pd.Series(data=model.feature_importances_, name=this_target, index=dataframe.columns)
			importances = pd.concat([importances, scores], axis=1)
		importances = importances.mean(axis=1).sort_values(ascending=False)
		
		number_of_features_to_keep = int(round(len(importances) * self.fraction))
		features_to_keep = importances.iloc[:number_of_features_to_keep].index
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)
		
class FeaturePredictivityRule(Rule):
	def __init__(self, fraction, omic, database):
		# TODO: Add check on fraction allowed values
		self.fraction = fraction
		self.omic = omic
		self.database = database
	
	def create_filter(self, dataset, target_dataframe):
		dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
		if len(target_dataframe.shape) == 1: ### utility to do this (takes a long time) on several targets?
			target_dataframe = target_dataframe.to_frame()
# 		index = target_dataframe.index[target_df.apply(np.isnan)]   ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost so we need to drop them
# 		to_drop = index.values.tolist()
# 		dataframe = dataframe.drop(to_drop)
# 		target_dataframe = target_dataframe.drop(to_drop)
		predictivities = pd.DataFrame()
		for this_target in target_dataframe.columns:
			model = xgb.XGBRegressor(max_depth=4, n_estimators=100, colsample_bytree = 0.5) ### deeper?
			model.fit(dataframe, target_dataframe[this_target])
			predicted = model.predict(dataframe)
			base_error = mean_squared_error(target_dataframe[this_target], predicted)
			base_df = dataframe.copy()
			this_target_predictivity = []
			for this_feature in dataframe.columns:
				shuffled_df = base_df.copy()
				shuffled_df[this_feature] = np.random.permutation(shuffled_df[this_feature].values)
				model = xgb.XGBRegressor(max_depth=4, n_estimators=100, colsample_bytree = 0.5) ### deeper?
				model.fit(dataframe, target_dataframe[this_target])
				shuffled_predicted = model.predict(shuffled_df)
				shuffled_error = mean_squared_error(target_dataframe[this_target], shuffled_predicted)
				this_target_predictivity.append(base_error - shuffled_error)
			target_predictivity = pd.Series(data=this_target_predictivity, name=this_target, index=dataframe.columns)
			predictivities = pd.concat([predictivities, target_predictivity], axis=1)
		predictivities = predictivities.mean(axis=1).sort_values(ascending=True)
		
		number_of_features_to_keep = int(round(len(predictivities) * self.fraction))
		features_to_keep = predictivities.iloc[:number_of_features_to_keep].index
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)
		





















