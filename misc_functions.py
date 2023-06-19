# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:16:48 2023
Miscellaneous functions
@author: apurva.badkas
"""

import pandas as pd
import glob


def cols_with_nans(data):
    """
    Identify columns with nans if any
    """
    
    df_with_nans = data.dataframe[data.dataframe.isna().any(axis=1)]
    nan_cols = df_with_nans.columns[df_with_nans.isna().any()]
    nan_df = df_with_nans[nan_cols]
    return nan_df

def merge_discretized(folder):
    """" 
    Merge discretized datafiles
    """
    disc_files = glob.glob('{folder}\*_discretized*.csv'.format(folder=folder))
    dfs = [pd.read_csv(x, index_col=0) for x in disc_files]
    df_concat = pd.concat(dfs, axis=1)    
    return df_concat


concat_df = merge_discretized('data')
concat_df.to_csv('data/Combined_discretized_data_T_5.csv')