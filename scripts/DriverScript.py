# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""

from data_characterization import explore_shape, reduce_mem_usage, show_me_the_data
from data_preprocessing import reformat_drugs, eliminate_sparse_data
from outputs_engineering import *
from feature_engineering import *
from data_modeling import *

import pandas as pd
import seaborn as sns
sns.set(context='talk')

###############################################################################
# Import the data
dfProt = pd.read_csv('CCLE_RPPA_20181003.csv')
dfDrug = pd.read_csv('CCLE_NP24.2009_Drug_data_2015.02.24.csv')

dfProt = reduce_mem_usage(dfProt)
dfDrug = reduce_mem_usage(dfDrug)

# Check the data
nCellLinesRPPA, nFeatures, percentMissingRPPA= explore_shape(dfProt)
dfDrug = reformat_drugs(dfDrug)
nCellLinesDrug, nOutputs, percentMissingDrug = explore_shape(dfDrug)

# Reshape the drug info
dfDrug = reformat_drugs(dfDrug)

# Remove obviously unusable data (lines or columns having less than x% of data)
dfProt = eliminate_sparse_data(dfProt, colThreshold = 0.8, lineThreshold = 0.8)
dfDrug = eliminate_sparse_data(dfDrug, colThreshold = 0.8, lineThreshold = 0.8)

# Visualize the data
show_me_the_data(dfProt)
show_me_the_data(dfDrug)




