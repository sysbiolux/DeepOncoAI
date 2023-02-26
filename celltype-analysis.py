# analyse 10x nested x-val data by cell type:


import numpy as np
import pandas as pd
from config import Config

from DBM_toolbox.data_manipulation.data_utils import pickle_objects, unpickle_objects

final_results = unpickle_objects('FINAL_results_2023-02-20-12-35-21-577662.pkl')
config = Config("testall/config_paper.yaml")


drugs_list = list(final_results.keys())

tumors_list = [
        "PROSTATE",
        "STOMACH",
        "URINARY",
        "NERVOUS",
        "OVARY",
        "HAEMATOPOIETIC",
        "KIDNEY",
        "THYROID",
        "SKIN",
        "SOFT_TISSUE",
        "SALIVARY",
        "LUNG",
        "BONE",
        "PLEURA",
        "ENDOMETRIUM",
        "BREAST",
        "PANCREAS",
        "AERODIGESTIVE",
        "LARGE_INTESTINE",
        "GANGLIA",
        "OESOPHAGUS",
        "FIBROBLAST",
        "CERVIX",
        "LIVER",
        "BILIARY",
        "SMALL_INTESTINE",
    ]

exp = [x + '_N' for x in drugs_list]

df_all = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
df_pos = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
df_neg = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
df_mcc = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
df_prec = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
df_recall = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)

for drug in drugs_list:
    for tumor in tumors_list:
        print(f'drug: {drug}, tumor: {tumor}')
        res = final_results[drug].loc[:, [drug, 'pred2_RFC']]
        res = res[res.index.str.contains(tumor)]
        N_pos = sum(res.loc[:, drug] == 1)
        N_neg = sum(res.loc[:, drug] == 0)
        N_all = N_pos + N_neg
        if N_all >= 10:
            res['sum'] = res.sum(axis=1)
            res['tp'] = res['sum'] >= 1.5
            res['tn'] = res['sum'] <= 0.5
            res['fp'] = (res['sum'] > 0.5) & (res['sum'] <= 1)
            res['fn'] = (res['sum'] > 1) & (res['sum'] <= 1.5)
            res['pp'] = res['pred2_RFC'] > 0.5
            res['pn'] = res['pred2_RFC'] < 0.5
            df_pos.loc[tumor, drug +'_N'] = N_pos
            df_neg.loc[tumor, drug +'_N'] = N_neg
            df_all.loc[tumor, drug +'_N'] = N_all
            if N_pos > 0:
                df_pos.loc[tumor, drug] = res.loc[:, 'tp'].sum().sum() / N_pos
            if N_neg > 0:
                df_neg.loc[tumor, drug] = res.loc[:, 'tn'].sum().sum() / N_neg
            if res.loc[:, 'pp'].astype(int).sum() > 0:
                df_prec.loc[tumor, drug] = res.loc[:, 'tp'].sum().sum().astype(int) / (res.loc[:, 'pp'].astype(int)).sum().sum()
                df_recall.loc[tumor, drug] = res.loc[:, 'tp'].sum().sum().astype(int) / (res.loc[:, 'tp'].astype(int) + res.loc[:, 'fn'].astype(int)).sum().sum()
                df_all.loc[tumor, drug] = res.loc[:, ['tp', 'tn']].sum().sum().astype(int) / N_all
                # df_mcc.loc[tumor, drug] = res['tp']*res['tn'] - res['fp']*res['fn'] / np.sqrt()
df_pos = df_pos.dropna(how='all')
df_neg = df_neg.dropna(how='all')
df_all = df_all.dropna(how='all')
df_prec = df_prec.dropna(how='all')
df_recall = df_recall.dropna(how='all')

df_pos.to_csv('FINAL_celltype_results_pos.csv')
df_neg.to_csv('FINAL_celltype_results_neg.csv')
df_all.to_csv('FINAL_celltype_results_all.csv')
df_prec.to_csv('FINAL_celltype_results_prec.csv')
df_recall.to_csv('FINAL_celltype_results_recall.csv')

#####################################################################
#####################################################################

