# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:16:23 2020

@author: sebde
"""

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def return_PPV_NPV(y, y_proba, dfTissue, classes, target = 'target'):
    TissueResults = pd.DataFrame(data = None, index = classes, columns = ['BalAcc', 'TP', 'TN', 'FP', 'FN'])
    y_sensitivity = y.to_frame()
    y_sensitivity['prediction']= [int(x > 0.5) for x in y_proba]
    df_results = pd.merge(dfTissue.rename('Tissue'), y_sensitivity, left_index=True, right_index=True )
    for this_tissue in TissueResults.index:
        df_tmp = df_results[df_results['Tissue'] == this_tissue]
        if not df_tmp.empty:
            TissueResults.loc[this_tissue, ['TN', 'FP', 'FN', 'TP']] = confusion_matrix(df_tmp[target], df_tmp['prediction']).ravel()
            TissueResults.loc[this_tissue, ['BalAcc'] ]= balanced_accuracy_score(df_tmp[target], df_tmp['prediction'])
    TissueResults.dropna(inplace=True)
    TissueResults = TissueResults[TissueResults.sum(axis = 1)>=10] #at least 10 cell lines
    TissueResults['PPV'] = TissueResults['TP']/ (TissueResults['TP'] + TissueResults['FP'] + 1E-15) #added constant to avoid division by zero
    TissueResults['NPV'] = TissueResults['TN']/ (TissueResults['TN'] + TissueResults['FN']+ 1E-15)
    
    return TissueResults

