import numpy as np
import pandas as pd
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
		
		print(importances)
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
			print(base_error)
			base_df = dataframe.copy()
			this_target_predictivity = []
			for this_feature in dataframe.columns:
				shuffled_df = base_df.copy()
				shuffled_df[this_feature] = np.random.permutation(shuffled_df[this_feature].values)
				model = xgb.XGBRegressor(max_depth=4, n_estimators=100, colsample_bytree = 0.5) ### deeper?
				model.fit(dataframe, target_df[this_target])
				shuffled_predicted = model.predict(shuffled_df)
				shuffled_error = mean_squared_error(target_df[this_target], shuffled_predicted)
				print(shuffled_error)
				this_target_predictivity.append(base_error - shuffled_error)
			print(this_target_predictivity, this_target, dataframe.columns)
			target_predictivity = pd.Series(data=this_target_predictivity, name=this_target, index=dataframe.columns)
			predictivities = pd.concat([predictivities, target_predictivity], axis=1)
			print(predictivities)
		predictivities = predictivities.mean(axis=1).sort_values(ascending=True)
		
		number_of_features_to_keep = int(round(len(predictivities) * self.fraction))
		features_to_keep = predictivities.iloc[:number_of_features_to_keep].index
		return KeepFeaturesFilter(features=features_to_keep, omic=self.omic, database=self.database)
		





















