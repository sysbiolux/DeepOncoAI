# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""

from data_characterization import explore_shape, reduce_mem_usage, show_me_the_data
from data_preprocessing import reformat_drugs, eliminate_sparse_data, impute_missing_data, remove_outliers
from outputs_engineering import transform_zscores, get_drug_response
from feature_engineering import add_polynomials, categorize_data
from data_modeling import get_models, make_pipeline, evaluate_model, robust_evaluate_model, evaluate_models, summarize_results


import pandas as pd
import seaborn as sns
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
dfProt = pd.read_csv('CCLE_RPPA_20181003.csv')
dfDrug = pd.read_csv('CCLE_NP24.2009_Drug_data_2015.02.24.csv')

dfProt = reduce_mem_usage(dfProt)
dfDrug = reduce_mem_usage(dfDrug)

# Check the data
nCellLinesRPPA, nFeatures, percentMissingRPPA = explore_shape(dfProt)
nCellLinesDrug, nOutputs, percentMissingDrug = explore_shape(dfDrug)


# Reshape the drug info
dfDrug = reformat_drugs(dfDrug)

# Remove obviously unusable data (lines or columns having less than x% of data)
dfProt = eliminate_sparse_data(dfProt, colThreshold = 0.8, lineThreshold = 0.8)
dfDrug = eliminate_sparse_data(dfDrug, colThreshold = 0.8, lineThreshold = 0.8)

# Visualize the data
show_me_the_data(dfProt)
show_me_the_data(dfDrug)

# Impute data if necessary
dfProt_I = impute_missing_data(dfProt)
dfDrug_I = impute_missing_data(dfDrug)

# Remove outliers

dfDrug_I_O = remove_outliers(dfDrug_I)
dfProt_I_O = remove_outliers(dfDrug_I)

# Get Outputs as z-scores
drugZScores = transform_zscores(dfDrug_I_O)

# Get Outputs as Resistant (-1), Sensitive (1), Intermediate (0)
thresholdR = -2
thresholdS = 2
drugResponse = get_drug_response(drugZScores, thresholdR, thresholdS)

# Outputs discretization
drugResponseCategorical = categorize_data(drugResponse)













# Add complexity with polynomial combinations
polynomialDegree = 2
dfExtended = add_polynomials(dfProt_I_O, degree = polynomialDegree)








coltodrop = list(df_norm_scaled.iloc[:,0:214].columns)
dfdropped = df_norm_scaled.drop(columns = coltodrop)
colnames = list(dfdropped.columns)












