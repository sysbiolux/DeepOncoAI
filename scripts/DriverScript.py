# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""

from data_characterization import explore_shape, reduce_mem_usage, show_me_the_data
from data_preprocessing import reformat_drugs, eliminate_sparse_data, impute_missing_data, remove_outliers, get_PCA, get_TSNE, select_top_features
from outputs_engineering import transform_zscores, get_drug_response
from feature_engineering import add_polynomials, categorize_data
from data_modeling import get_regression_models, get_classification_models, evaluate_models, summarize_results
from results_analysis import plot_confusion_matrix, plot_roc, plot_decision_boundary

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from sklearn import metrics
import logging
import time
from datetime import datetime

import numba
#Acceleration
@numba.jit
def f(x):
	return x
@numba.njit
def f(x):
	return x

now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")
logging.basicConfig(filename = 'debug_'+timestamp+'.log',level=logging.DEBUG)
logging.debug('log spotcheck')
sns.set(context='talk')


###############################################################################

Goal = 'classification'
Targets = ['IC50', 'ActArea', 'Amax']
polynomialDegree = 0

###############################################################################
# Import the data
dfProt_orig = pd.read_csv('CCLE_RPPA_20181003.csv')
dfProt_orig = dfProt_orig.set_index('Unnamed: 0')
dfDrug_orig = pd.read_csv('CCLE_NP24.2009_Drug_data_2015.02.24.csv')

dfProt = reduce_mem_usage(dfProt_orig)
dfDrug = reduce_mem_usage(dfDrug_orig)

# Check the data
nCellLinesRPPA, nFeatures, percentMissingRPPA = explore_shape(dfProt)
nCellLinesDrug, nOutputs, percentMissingDrug = explore_shape(dfDrug)


# Reshape the drug info
dfDrug = reformat_drugs(dfDrug)

# Remove obviously unusable data (lines or columns having less than x% of data)
dfProt = eliminate_sparse_data(dfProt, colThreshold = 0.8, rowThreshold = 0.1)
dfDrug = eliminate_sparse_data(dfDrug, colThreshold = 0.8, rowThreshold = 0.1)

##sample in the dataset
#dfProt = dfProt.sample(n=200, random_state=42, axis=0)
#dfProt = dfProt.sample(n=100, random_state=42, axis=1)
#dfDrug = dfDrug.sample(n=200, random_state=42, axis=0)



#Choosing the right target: 'IC50', 'Amax', 'EC50', or 'ActArea'
for thisTarget in Targets:
	cols = [col for col in dfDrug.columns if thisTarget in col]
	dfDrugX = dfDrug[cols]


	# Impute data if necessary
	dfProt_I = impute_missing_data(dfProt)
	dfDrug_I = impute_missing_data(dfDrugX)

	# Visualize the data
	#show_me_the_data(dfProt_I)
	#show_me_the_data(dfDrug_I)

	# Remove outliers

	dfDrug_I_O = remove_outliers(dfDrug_I)
	dfProt_I_O = remove_outliers(dfProt_I)

	# Normalize the features 0-1
	x = dfProt_I_O.values #returns a numpy array
	scaler = pp.MinMaxScaler()
	x_scaled = scaler.fit_transform(x)
	dfProt_I_O_N = pd.DataFrame(data = x_scaled, index = dfProt_I_O.index, columns = dfProt_I_O.columns)

	if thisTarget == 'IC50':
		dfDrug_I_O = -dfDrug_I_O

	# Normalize the targets 0-1
	x = dfDrug_I_O.values #returns a numpy array
	scaler = pp.MinMaxScaler()
	x_scaled = scaler.fit_transform(x)
	dfDrug_I_O_N = pd.DataFrame(data = x_scaled, index = dfDrug_I_O.index, columns = dfDrug_I_O.columns)

	show_me_the_data(dfDrug_I_O_N)

	# Get Outputs as z-scores
	drugZScores = transform_zscores(dfDrug_I_O)

	# Get Outputs as Resistant (-1), Sensitive (1), Intermediate (0)
	for strategy in range(4):
		if strategy == 0:
			thresholdR = np.mean(dfDrug_I_O_N, axis = 0)
			thresholdS = thresholdR
		if strategy == 1:
			thresholdR = dfDrug_I_O_N.quantile(0.66)
			thresholdS = dfDrug_I_O_N.quantile(0.33)
		if strategy == 2:
			thresholdR = dfDrug_I_O_N.quantile(0.75)
			thresholdS = dfDrug_I_O_N.quantile(0.25)
		if strategy == 3:
			thresholdR = dfDrug_I_O_N.quantile(0.8)
			thresholdS = dfDrug_I_O_N.quantile(0.2)

		drugResponse = get_drug_response(dfDrug_I_O_N, thresholdR, thresholdS)

		# Remove intermediates
		drugResponse = drugResponse[drugResponse != 0]

		# Outputs discretization
		drugResponseCategorical = categorize_data(drugResponse)


		# Add complexity with polynomial combinations
		dfExtended = add_polynomials(dfProt_I_O_N, degree = polynomialDegree)

		dfMerged = pd.merge(dfExtended, drugResponse, left_index = True, right_index = True)
		dfMergedZscores = pd.merge(dfExtended, drugZScores, left_index = True, right_index = True)
		dfPredictors = dfMerged[dfExtended.columns]
		dfTargets = dfMerged[drugResponse.columns]
		dfTargetsZscores = dfMergedZscores[drugZScores.columns]


		#Dimension reduction
		targets = [-1, 0, 1]
		colors = ['r', 'b', 'g']
		df_PCs = get_PCA(dfPredictors, n_components = min(min(dfPredictors.shape), 20))

#		for drug in dfTargets.columns:
#			fig, ax = plt.subplots()
#			ax.set_title(drug)
#			for target, color in zip(targets, colors):
#				idx = dfTargets[drug].values == target
#				toPlot = df_PCs.iloc[idx]
#				ax.scatter(toPlot.iloc[:,0], toPlot.iloc[:,1], c = color, s = 50)
#			ax.legend(['R', 'I', 'S'])

		df_TSNEs = get_TSNE(dfPredictors, n_components = 3)

#		for drug in dfTargets.columns:
#			fig, ax = plt.subplots()
#			ax.set_title(drug)
#			for target, color in zip(targets, colors):
#				idx = dfTargets[drug].values == target
#				toPlot = df_TSNEs.iloc[idx]
#				ax.scatter(toPlot.iloc[:,0], toPlot.iloc[:,1], c = color, s = 50)
#			ax.legend(['R', 'I', 'S'])

		# Merge reduced dimension with other predictors
		dfPredictors = pd.merge(dfPredictors, pd.merge(df_PCs, df_TSNEs, left_index = True, right_index = True), left_index = True, right_index = True)

		if Goal == 'regression':
			y = dfTargetsZscores
		if Goal == 'classification':
			y = dfTargets
		X = dfPredictors.copy()

		# Perform rough feature selection
		X = select_top_features(X, y, threshold=0)

		#############################################################################
		# Modeling data
		for thisCol in dfTargetsZscores.columns:
			X_train, X_test, y_train, y_test = train_test_split(X, y[thisCol], test_size=0.2, random_state=123)
			index = y_train.index[y_train.apply(np.isnan)]
			todrop = index.values.tolist()
			X_train = X_train.drop(todrop)
			y_train = y_train.drop(todrop)
			index = y_test.index[y_test.apply(np.isnan)]
			todrop = index.values.tolist()
			X_test = X_test.drop(todrop)
			y_test = y_test.drop(todrop)


			print('Investigating %s' % (thisCol))
			# get model list
			if Goal == 'regression':
				models = get_regression_models(depth = 1)
				# evaluate models
				results, predicted = evaluate_models(X_train, y_train, models, X_test, metric='neg_mean_squared_error')
			if Goal == 'classification':
				models = get_classification_models(depth = 1)
				# evaluate models
				results, predicted = evaluate_models(X_train, y_train, models, X_test, metric='balanced_accuracy')
			# summarize results
			thisNames, thisScores, thisPredict = summarize_results(results, predicted, y_test, thisCol)

			np.set_printoptions(precision=2)
			fig, ax = plt.subplots(3,4, figsize=(30, 50))
			#plt.tight_layout()
			ax = ax.ravel()
			fig.suptitle('Top-3 algos for %s with strategy %d' % (thisCol, strategy))
			print(thisNames[0])
			plot_confusion_matrix(y_test, thisPredict[0], classes=['R', 'S'], ax=ax[0], title=('Confusion matrix, %s' % (thisNames[0])))
			plot_confusion_matrix(y_test, thisPredict[0], classes=['R', 'S'], ax=ax[1], normalize=True, title=('Normalized Confusion matrix, %s' % (thisNames[0])))
			plot_roc(y_test, X_test, y_train, X_train, thisNames, models, item=0, ax=ax[2])
			plot_decision_boundary(y_test, X_test, y_train, X_train, thisNames, models, item=0, ax=ax[3])
			print(thisNames[1])
			plot_confusion_matrix(y_test, thisPredict[1], classes=['R', 'S'], ax=ax[4], title=('Confusion matrix, %s' % (thisNames[1])))
			plot_confusion_matrix(y_test, thisPredict[1], classes=['R', 'S'], ax=ax[5], normalize=True, title=('Normalized Confusion matrix, %s' % (thisNames[1])))
			plot_roc(y_test, X_test, y_train, X_train, thisNames, models, item=1, ax=ax[6])
			plot_decision_boundary(y_test, X_test, y_train, X_train, thisNames, models, item=1, ax=ax[7])
			print(thisNames[2])
			plot_confusion_matrix(y_test, thisPredict[2], classes=['R', 'S'], ax=ax[8], title=('Confusion matrix, %s' % (thisNames[2])))
			plot_confusion_matrix(y_test, thisPredict[2], classes=['R', 'S'], ax=ax[9], normalize=True, title=('Normalized Confusion matrix, %s' % (thisNames[2])))
			plot_roc(y_test, X_test, y_train, X_train, thisNames, models, item=2, ax=ax[10])
			plot_decision_boundary(y_test, X_test, y_train, X_train, thisNames, models, item=2, ax=ax[11])
			plt.show()
			plt.savefig('Best3Models_'+thisCol+'.pdf')




