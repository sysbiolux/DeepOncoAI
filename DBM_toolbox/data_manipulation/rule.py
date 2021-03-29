import numpy as np
import pandas as pd
import random
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
	def __init__(self, completeness_threshold, omic, database):
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
	

	def create_filter(self, dataset):
		
		def get_features(dataframe, corr_threshold):
			names = dataframe.columns
			print('computing correlation matrix')
			corr_matrix = dataframe.corr().abs()
			corr_array = corr_matrix.to_numpy()
			np.fill_diagonal(corr_array, 0)
			clean = (corr_array.max() < self.correlation_threshold)
			n_features_start = dataframe.shape[1]
			
			while not clean: #TODO: this takes time, look at efficiency
				print('removing correlated features')
				sum_correlations = corr_matrix.sum()
				ind = np.unravel_index(np.argmax(corr_array), corr_array.shape)
				value_to_remove = np.max([sum_correlations[names[ind[0]]], sum_correlations[names[ind[1]]]])
				feature_to_remove = sum_correlations[sum_correlations==value_to_remove].index
				dataframe = dataframe.drop(feature_to_remove, axis=1)
				names = dataframe.columns
				corr_matrix = dataframe.corr().abs()
				corr_array = corr_matrix.to_numpy()
				np.fill_diagonal(corr_array, 0)
				clean = (corr_array.max() < self.correlation_threshold)
			
			features_to_keep = dataframe.columns
			n_features_removed = n_features_start - dataframe.shape[1]
			
			print('Removed ' + str(n_features_removed) + ' highly correlated features from a total of ' + str(n_features_start))
			return features_to_keep
		
		def shuffle_columns(dataframe, seed=42):
			col_names = list(dataframe.columns)
			random.seed(seed)
			random.shuffle(col_names)
			dataframe = dataframe[col_names]
			
			return dataframe
		
		dataframe = dataset.to_pandas(omic=self.omic)
		if len(dataframe.columns) < 1000:
			features_to_keep = get_features(dataframe, self.correlation_threshold)
		else: # cannot compute pandas.corr() for large matrices
			print('large dataset')
			keep_on_trying = True
			while keep_on_trying:
				print('shuffling')
				dataframe = shuffle_columns(dataframe=dataframe)
				keep_on_trying = False
				features_to_keep = []
				n_chunks = round(len(dataframe.columns)/1000 + 0.5)
				starts = [x * 1000 for x in (range(n_chunks))]
				stops = [x + 999 for x in starts]
				stops[-1] = len(dataframe.columns)
				for count, col_start in enumerate(starts):
					print('chunk ' + str(count))
					col_stop = stops[count]
					mini_dataframe = dataframe.iloc[:, col_start:col_stop + 1]
					features_to_keep.append(get_features(mini_dataframe, self.correlation_threshold))
				
				new_dataframe = dataframe[features_to_keep[0]]
				for chunk in range(1, n_chunks):
					print('merging chunks')
					new_dataframe.merge(dataframe[features_to_keep[chunk]], left_index=True, right_index=True)
				keep_on_trying = (new_dataframe.size != dataframe.size)
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)
	
class FeatureImportanceRule(Rule):
	def __init__(self, fraction, omic, database):
		# TODO: Add check on fraction allowed values
		self.fraction = fraction
		self.omic = omic
		self.database = database
		
	def create_filter(self, dataset, target_df):
		dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
		if len(target_df.shape) == 1:
			target_df = target_df.to_frame()
# 		index = target_df.index[target_df.apply(np.isnan)]  ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost
# 		to_drop = index.values.tolist()
# 		dataframe = dataframe.drop(to_drop)
# 		target_df = target_df.drop(to_drop)
		
		importances = pd.DataFrame()
		for this_target in target_df.columns:
			model = xgb.XGBClassifier(max_depth=4, n_estimators=100, colsample_bytree = 0.5) ### deeper?
			model.fit(dataframe, target_df)
			scores = pd.Series(data=model.feature_importances_, name=this_target, index=dataframe.columns)
			importances = pd.concat([importances, scores], axis=1)
		importances = importances.mean(axis=1).sort_values(ascending=False)
		
# 		print(importances)
		number_of_features_to_keep = int(round(len(importances) * self.fraction))
		features_to_keep = importances.iloc[:number_of_features_to_keep].index
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)
		
class FeaturePredictivityRule(Rule):
	def __init__(self, fraction, omic, database):
		# TODO: Add check on fraction allowed values
		self.fraction = fraction
		self.omic = omic
		self.database = database
	
	def create_filter(self, dataset, target_df):
		dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
		if len(target_df.shape) == 1: ### utility to do this (takes a long time) on several targets?
			target_df = target_df.to_frame()
# 		index = target_df.index[target_df.apply(np.isnan)]   ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost so we need to drop them
# 		to_drop = index.values.tolist()
# 		dataframe = dataframe.drop(to_drop)
# 		target_df = target_df.drop(to_drop)
		predictivities = pd.DataFrame()
		for this_target in target_df.columns:
			model = xgb.XGBRegressor(max_depth=4, n_estimators=100, colsample_bytree = 0.5) ### deeper?
			model.fit(dataframe, target_df[this_target])
			predicted = model.predict(dataframe)
			base_error = mean_squared_error(target_df[this_target], predicted) #balanced_accuracy_score(target_df[this_target], predicted)
			base_df = dataframe.copy()
			this_target_predictivity = []
			for this_feature in dataframe.columns:
				shuffled_df = base_df.copy()
				shuffled_df[this_feature] = np.random.permutation(shuffled_df[this_feature].values)
				model = xgb.XGBRegressor(max_depth=4, n_estimators=100, colsample_bytree = 0.5) ### deeper?
				model.fit(dataframe, target_df[this_target])
				shuffled_predicted = model.predict(shuffled_df)
				shuffled_error = mean_squared_error(target_df[this_target], shuffled_predicted)
				this_target_predictivity.append(base_error - shuffled_error)
			target_predictivity = pd.Series(data=this_target_predictivity, name=this_target, index=dataframe.columns)
			predictivities = pd.concat([predictivities, target_predictivity], axis=1)
		predictivities = predictivities.mean(axis=1).sort_values(ascending=True)
		
		number_of_features_to_keep = int(round(len(predictivities) * self.fraction))
		features_to_keep = predictivities.iloc[:number_of_features_to_keep].index
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)
		





















