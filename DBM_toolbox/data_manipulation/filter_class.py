import pandas as pd
from DBM_toolbox.data_manipulation.dataset_class import Dataset

class KeepFeaturesFilter:
	def __init__(self, features, omic, database):
		self.features = features
		self.omic = omic
		self.database = database

	def apply(self, dataset):
		# TODO: this takes too much time!
		dataframe = dataset.dataframe
		features_to_keep = []
		retained_omic = pd.Series()
		retained_database = pd.Series()
		for this_feature in dataframe.columns:
			if dataset.omic[this_feature] != self.omic:
				features_to_keep.append(this_feature)
				retained_omic = pd.concat([retained_omic, pd.Series(dataset.omic[this_feature], index=[this_feature])])
				retained_database = pd.concat([retained_database, pd.Series(dataset.database[this_feature], index=[this_feature])])
			else:
				if this_feature in self.features:
					features_to_keep.append(this_feature)
					retained_omic = pd.concat([retained_omic, pd.Series(dataset.omic[this_feature], index=[this_feature])])
					retained_database = pd.concat([retained_database, pd.Series(dataset.database[this_feature], index=[this_feature])])
		filtered_dataframe = dataframe[features_to_keep]
		return Dataset(dataframe=filtered_dataframe, omic=retained_omic, database=retained_database)

	def __repr__(self):
		return f'KeepFeaturesFilter({self.features}, {self.omic})'


class KeepDenseRowsFilter:
	def __init__(self, completeness_threshold):
		self.completeness_threshold = completeness_threshold

	def apply(self, dataset):
		dataframe = dataset.dataframe
		completeness = 1 - (dataframe.isna().mean(axis = 1))
		samples_to_keep = completeness[completeness >= self.completeness_threshold].index
		filtered_dataframe = dataframe.loc[samples_to_keep]
		return Dataset(dataframe=filtered_dataframe, omic=dataset.omic, database=dataset.database)

