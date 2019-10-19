# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""
import numpy as np
import pandas as pd


def eliminate_sparse_data(df, colThreshold = 0.5, rowThreshold = 0.5):
	"""drops the columns and lines that do not satisfy the thresholds in term of
	presence of data"""
	#df_orig = df.copy()
	colFil = 1 - df.isna().mean()
	rowFil = 1 - df.isna().mean(axis = 1)

	df = df.loc[rowFil > rowThreshold, :]
	df = df.loc[:,colFil > colThreshold]

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
	if method == 'average':
		df = df.fillna(df.mean())

	return df

def remove_outliers(df, method = 'normal', param = 0.05):
	"""removes outliers from each column independently according to different methods"""
	#TODO:

	return df

def get_PCA(df, n_components = 2):
	"""adds columns corresponding to the PCA components of the dataset"""
	from sklearn.decomposition import PCA

	pca = PCA(n_components = n_components)
	principalComponents = pca.fit_transform(df)
	colList = []
	for n in range(1, n_components+1):
		colList.append('PC'+str(n))

	df_PCs = pd.DataFrame(data = principalComponents, index = df.index, columns = colList)
	print(pca.explained_variance_ratio_)

	return df_PCs

def get_TSNE(df, n_components = 2):
	from sklearn.manifold import TSNE

	tsne = TSNE(n_components = n_components, verbose=1)
	tsneComponents = tsne.fit_transform(df)
	colList = []
	for n in range(1, n_components+1):
		colList.append('TSNE'+str(n))

	df_TSNEs = pd.DataFrame(data = tsneComponents, index = df.index, columns = colList)

	return df_TSNEs

def select_top_features(X, y, threshold=0):
	# Perform rough feature selection
	import xgboost as xgb
	from sklearn.metrics import balanced_accuracy_score
	from sklearn.model_selection import train_test_split

	featureScores = pd.DataFrame()
	for col in y.columns:
		# split data into train and test sets
		X_train, X_test, y_train, y_test = train_test_split(X, y[col], test_size=0.25, random_state=42)
		index = y_train.index[y_train.apply(np.isnan)]
		todrop = index.values.tolist()
		X_train = X_train.drop(todrop)
		y_train = y_train.drop(todrop)
		index = y_test.index[y_test.apply(np.isnan)]
		todrop = index.values.tolist()
		X_test = X_test.drop(todrop)
		y_test = y_test.drop(todrop)
		# fit model on all training data
		model = xgb.XGBClassifier(max_depth=5, n_estimators=200)
		model.fit(X_train, y_train)
		# make predictions for test data and evaluate
		y_pred = model.predict(X_test)
		predictions = [round(value) for value in y_pred]
		accuracy = balanced_accuracy_score(y_test, predictions)
		print("Balanced accuracy: %.2f%%" % (accuracy * 100.0))
		if featureScores.empty:
			featureScores = pd.Series(data=model.feature_importances_, name=col)
		else:
			featureScores = pd.concat([featureScores, pd.Series(data=model.feature_importances_, name=col)], axis=1)

	meanFeatureScores = np.mean(featureScores, axis=1)
	toDrop = meanFeatureScores <= threshold
	Xd = X[X.columns[~toDrop]]

	return Xd









