# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""

from data_characterization import *
from data_preprocessing import *
from outputs_engineering import *
from feature_engineering import *
from data_modeling import *

import pandas as pd
import seaborn as sns
sns.set(context='talk')

###############################################################################
# Import the data #
dfProt = pd.read_csv('CCLE_RPPA_20181003.csv')
dfDrug = pd.read_csv('CCLE_NP24.2009_Drug_data_2015.02.24.csv')

dfProt = reduce_mem_usage(dfProt)
dfDrug = reduce_mem_usage(dfDrug)

# Check the data #
nCellLinesRPPA, nFeatures, percentMissingRPPA= explore_shape(dfProt)
dfDrug = reformat_drugs(dfDrug)
nCellLinesDrug, nOutputs, percentMissingDrug = explore_shape(dfDrug)




