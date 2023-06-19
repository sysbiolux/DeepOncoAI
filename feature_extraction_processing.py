# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:52:48 2023

@author: apurva.badkas
"""

import pandas as pd
import glob

#APPPARTIAL_FI_med_ZD-6474_APPROXCURRBET_ET
#AVNPARTIAL_FI_med_17-AAG_AVNEIGHBOUR_Ada
#BETPARTIAL_FI_med_17-AAG_BETWEENNESS_SVC
#CLIPARTIAL_FI_med_17-AAG_CLIQUENO_Logistic
#CLOPARTIAL_FI_med_17-AAG_CLOSENESS_SVC
#CONPARTIAL_FI_med_17-AAG_CONSTRAINT_EN
#DEGPARTIAL_FI_med_17-AAG_DEGREECENT_Ada
#ECCPARTIAL_FI_med_AEW541_ECCENTRICITY_ET
#HARPARTIAL_FI_med_AEW541_HARMONIC_Logistic
#INFPARTIAL_FI_med_AEW541_INFORMATION_ET
#PAGPARTIAL_FI_med_17-AAG_PAGERANK_ET
#SUBPARTIAL_FI_med_AZD6244_SUBGRAPH_RFC
#SQUPARTIAL_FI_med_17-AAG_SQUARECLUSTERING_RFC
#EIGPARTIAL_FI_med_17-AAG_EIGENVECTOR_Logistic
#DISPARTIAL_FI_med_AZD0530_DISCRETIZED_ET

foldername = 'RNAAPPPARTIAL'
#drug = 'ZD-6474'
measures = ['RNA', 'APPROXCURRBET']

drugs = ['Lapatinib',
'Panobinostat',
'Paclitaxel',
'Irinotecan',
'PD-0325901',
'AZD6244',
'Nilotinib',
'AEW541',
'17-AAG',
'PHA-665752',
'Nutlin-3',
'AZD0530',
'PF2341066',
'L-685458',
'ZD-6474',
'Sorafenib',
'LBW242',
'PD-0332991',
'PLX4720',
'RAF265',
'TAE684',
'TKI258',
'Erlotinib']





def gene_frequency(files, drug, meas):
    cols =[]
    dfs = []
    df_cols = []
    
    for fil in files:
        df_raw = pd.read_csv(fil, index_col = 0)
        df_raw.index.name = fil.split('_')[-1]
        df = df_raw.reset_index()
        dfs.append(df)
        
    cols = [x.columns for x in dfs]    
    df_cols = [pd.DataFrame(y) for y in cols]
    coll_cols = []
    for col in df_cols:
        col.columns = col.iloc[0]    
        col_new = col.iloc[1:]
        coll_cols.append(col_new)
        
    
    ccc = []
    for y in coll_cols:
        ind = list(y.columns)
        if meas == 'RPPA':
            dx = y
        else:
            dx = y.iloc[:,0].str.split('_', expand=True)
        if dx.shape[1] == 1:
            dx = dx
        else:
            dx = dx.drop(columns=[1,2])
        dx.columns = ind
        dnew = dx.sort_values(by=ind)
        ccc.append(dnew)
    
    d = pd.concat(ccc, axis=1)    
    
    new_d = pd.melt(d)
    
    freq = pd.DataFrame(new_d.value_counts('value'))
    freq.columns = [drug+'_'+meas]
    
    return freq



all_freqs = []
for drug in drugs:
    for meas in measures:
        files = glob.glob(f'{foldername}/{foldername}_FI_med_{drug}_{meas}*')
        del files[0] # remove .tif file
        freq_df = gene_frequency(files, drug, meas)
        freq_df_name = list(freq_df.columns)[0]
        freq_df.to_csv(f'{foldername}/{freq_df_name}.csv')
        all_freqs.append(freq_df)






