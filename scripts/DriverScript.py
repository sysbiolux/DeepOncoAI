# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""

from data_characterization import explore_shape, reduce_mem_usage, show_me_the_data
from data_preprocessing import reformat_drugs, eliminate_sparse_data, impute_missing_data, remove_outliers
from outputs_engineering import transform_zscores, get_drug_response
from feature_engineering import add_polynomials, categorize_data
from data_modeling import get_regression_models, get_classification_models, make_pipeline, evaluate_model, robust_evaluate_model, evaluate_models, summarize_results


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

sns.set(context='talk')
import numba
#Acceleration
@numba.jit
def f(x):
	return x
@numba.njit
def f(x):
	return x


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

#Choosing the right target: 'IC50', 'Amax', 'EC50', or 'ActArea'
Target = 'ActArea'
cols = [col for col in dfDrug.columns if Target in col]
dfDrug = dfDrug[cols]




# Impute data if necessary
dfProt_I = impute_missing_data(dfProt)
dfDrug_I = impute_missing_data(dfDrug)

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

# Normalize the targets 0-1
x = dfDrug_I_O.values #returns a numpy array
scaler = pp.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
dfDrug_I_O_N = pd.DataFrame(data = x_scaled, index = dfDrug_I_O.index, columns = dfDrug_I_O.columns)


# Get Outputs as z-scores
drugZScores = transform_zscores(dfDrug_I_O)


# Get Outputs as Resistant (-1), Sensitive (1), Intermediate (0)
thresholdR = 0.4
thresholdS = 0.6
drugResponse = get_drug_response(dfDrug_I_O_N, thresholdR, thresholdS)





# Outputs discretization
drugResponseCategorical = categorize_data(drugResponse)


# Add complexity with polynomial combinations
polynomialDegree = 2
dfExtended = add_polynomials(dfProt_I_O_N, degree = polynomialDegree)

dfMerged = pd.merge(dfExtended, drugResponse, left_index = True, right_index = True)
dfMergedZscores = pd.merge(dfExtended, drugZScores, left_index = True, right_index = True)
dfPredictors = dfMerged[dfExtended.columns]
dfTargets = dfMerged[drugResponse.columns]
dfTargetsZscores = dfMergedZscores[drugZScores.columns]

#Dimension reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
targets = [-1, 0, 1]
colors = ['r', 'b', 'g']

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dfPredictors)
df_PCs = pd.DataFrame(data = principalComponents, index = dfPredictors.index, columns = ['PC1', 'PC2'])

for drug in dfTargets.columns:
	fig, ax = plt.subplots()
	ax.set_title(drug)
	for target, color in zip(targets, colors):
		idx = dfTargets[drug].values == target
		toPlot = df_PCs.iloc[idx]
		ax.scatter(toPlot.iloc[:,0], toPlot.iloc[:,1], c = color, s = 50)
		
	ax.legend(['R', 'I', 'S'])
	
print(pca.explained_variance_ratio_)


tsne = TSNE(n_components=2, verbose=1)
tsneComponents = tsne.fit_transform(dfPredictors)

df_TSNEs = pd.DataFrame(data = tsneComponents, index = dfPredictors.index, columns = ['TSNE1', 'TSNE2'])

for drug in dfTargets.columns:
	fig, ax = plt.subplots()
	ax.set_title(drug)
	for target, color in zip(targets, colors):
		idx = dfTargets[drug].values == target
		toPlot = df_TSNEs.iloc[idx]
		ax.scatter(toPlot.iloc[:,0], toPlot.iloc[:,1], c = color, s = 50)
		
	ax.legend(['R', 'I', 'S'])
	

# Merge reduced dimension with other predictors

dfPredictors = pd.merge(dfPredictors, pd.merge(df_PCs, df_TSNEs, left_index = True, right_index = True), left_index = True, right_index = True)

#############################################################################

from sklearn.model_selection import train_test_split
import numpy as np

for thisCol in dfTargetsZscores.columns:
	#Withold a test set
	X_train, X_test, y_train, y_test = train_test_split(dfPredictors, dfTargetsZscores[thisCol], test_size=0.2, random_state=42)
	index = y_train.index[y_train.apply(np.isnan)]
	todrop = index.values.tolist()
	X_train = X_train.drop(todrop)
	y_train = y_train.drop(todrop)
	index = y_test.index[y_test.apply(np.isnan)]
	todrop = index.values.tolist()
	X_test = X_test.drop(todrop)
	y_test = y_test.drop(todrop)
	# get model list
	models = get_regression_models(depth = 1)
	# evaluate models
	results, predicted = evaluate_models(X_train, y_train, models, X_test, metric='neg_mean_squared_error')
	# summarize results
	summarize_results(results, predicted, y_test, thisCol)


for thisCol in dfTargets.columns:
	#Withold a test set
	X_train, X_test, y_train, y_test = train_test_split(dfPredictors, dfTargets[thisCol], test_size=0.2, random_state=42)
	
	index = y_train.index[y_train.apply(np.isnan)]
	todrop = index.values.tolist()
	X_train = X_train.drop(todrop)
	y_train = y_train.drop(todrop)
	index = y_test.index[y_test.apply(np.isnan)]
	todrop = index.values.tolist()
	X_test = X_test.drop(todrop)
	y_test = y_test.drop(todrop)
	# get model list
	models = get_classification_models(depth = 1)
	# evaluate models
	results, predicted = evaluate_models(X_train, y_train, models, X_test, metric='accuracy')
	# summarize results
	summarize_results(results, predicted, y_test, thisCol)










