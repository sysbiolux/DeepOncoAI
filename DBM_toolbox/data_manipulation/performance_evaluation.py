# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from config import Config
from matplotlib import pyplot as plt
from DBM_toolbox.data_manipulation.data_utils import pickle_objects, unpickle_objects
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import glob
from pathlib import Path

##################################################################

def final_balanced_acc_plot(final_results, comb_name, results_folder):

    drugs_list = list(final_results.keys())
    
    tumors_list = [
            "NERVOUS",
            "SKIN",
            "LUNG",
            "BREAST",
            "LARGE_INTESTINE",
        ]
    
    exp = [x + '_N' for x in drugs_list]
    
    df_all = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
    df_pos = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
    df_neg = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
    df_prec = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
    df_recall = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
    df_ba = pd.DataFrame(index=tumors_list, columns=drugs_list + exp)
    
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
                    tpr = (res.loc[:, 'tp'].sum().sum().astype(int) / res.loc[:, ['tp', 'fn']].sum().sum().astype(int))
                    tnr = (res.loc[:, 'tn'].sum().sum().astype(int) / res.loc[:, ['tn', 'fp']].sum().sum().astype(int))
                    df_ba.loc[tumor, drug] = (tpr + tnr) / 2
    
    df_pos = df_pos.dropna(how='all')
    df_neg = df_neg.dropna(how='all')
    df_all = df_all.dropna(how='all')
    df_prec = df_prec.dropna(how='all')
    df_recall = df_recall.dropna(how='all')
    df_ba = df_ba.dropna(how='all')
    df_ba = df_ba.dropna(axis=1, how='all')
    
    df_pos.to_csv(f'{results_folder}/Topo_{comb_name}_celltype_results_pos.csv')
    df_neg.to_csv(f'{results_folder}/Topo_{comb_name}_celltype_results_neg.csv')
    df_all.to_csv(f'{results_folder}/Topo_{comb_name}_celltype_results_all.csv')
    df_prec.to_csv(f'{results_folder}/Topo_{comb_name}_celltype_results_prec.csv')
    df_recall.to_csv(f'{results_folder}/Topo_{comb_name}_celltype_results_recall.csv')
    df_ba.to_csv(f'{results_folder}/Topo_{comb_name}_celltype_results_bal-accuracy.csv')
    
    df_ba = df_ba.apply(pd.to_numeric, errors='coerce')
    
    fig, ax = plt.subplots(figsize=(30, 15))
    cmap = sns.cm.rocket_r
    sns.set(font_scale=1.4)
    sns.heatmap(df_ba, cmap=cmap, annot=True, fmt='.2f', linewidths=.1, ax=ax, annot_kws={
                    'fontsize': 16,
                    'fontweight': 'bold',
                    'fontfamily': 'serif'
                })
    #cmap=sns.color_palette("rocket", as_cmap=True)
    plt.xlabel('')
    plt.savefig(f'{results_folder}/{comb_name}_balanced_accuracy')
    plt.close()
    

#####################################################################
##################| ROC Curves

def plot_roc_curves(final_results, comb_name, results_folder, plot='FALSE'):
    hfont = {'fontname':'Times New Roman'}
    targets_names = list(final_results.keys())
    fprs = dict()
    tprs = dict()
    roc_aucs = dict()
    
    for target_name in targets_names:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        global_models = [x for x in final_results[target_name].columns if x.startswith('pred2')]
        plt.subplots(figsize=(8, 8))
        for global_model in global_models:
            y_score = final_results[target_name].loc[:, global_model]
            y_score = y_score[y_score.isna() == False]
            y_true = final_results[target_name].loc[:, target_name]
            y_true = y_true.loc[y_score.index]
            fpr[global_model], tpr[global_model], _ = roc_curve(y_true, y_score)
            roc_auc[global_model] = auc(fpr[global_model], tpr[global_model])
            if plot == 'TRUE':
            
                plt.plot(fpr[global_model], tpr[global_model], linewidth=3, label=f"AUC: {round(roc_auc[global_model], 3)}")
            plt.plot([0, 1], [0, 1], color="black", linestyle="--")
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=20, **hfont)
            plt.ylabel("True Positive Rate", fontsize=20, **hfont)
            plt.title(f"{target_name.split('_')[0]}",fontsize=20, **hfont)
            plt.legend(loc="lower right", fontsize=20)
            plt.savefig(f'{results_folder}/{comb_name}_{target_name}')
            plt.close()
    
        fprs[target_name] = fpr
        tprs[target_name] = tpr
        roc_aucs[target_name] = roc_auc
    
    roc_aucs_df = pd.DataFrame.from_dict(roc_aucs).T
    roc_aucs_df.to_csv(f'{results_folder}/{comb_name}_AUCS.csv')

###################################################################
### clustergrams

def plot_clustergram(features_file, comb_name, results_folder):
    #'f_test_topo2_features_2023-04-26-11-20-18-361678.pkl'
    fi = unpickle_objects(features_file)
    
    fi_names = fi['Lapatinib_ActArea'][0]['RFC'].index
    
    res_fi = pd.DataFrame(0, columns=fi_names, index=fi.keys())
    res_sd = pd.DataFrame(0, columns=fi_names, index=fi.keys())
    
    for target_name, target_dict in fi.items():
        these_means = pd.Series()
        for loop_number, loop_dict in target_dict.items():
            these_means = pd.concat([these_means, loop_dict['RFC']], axis=1)
        this_mean = these_means.mean(axis=1)
        this_sd = these_means.std(axis=1)
        res_fi.loc[target_name, :] = this_mean
        res_sd.loc[target_name, :] = this_sd
    colnames = res_fi.columns
    colnames = ['+'.join(x.split('_')[1:]) for x in colnames]
    indexnames = res_fi.index
    indexnames = [x.split('_')[0] for x in indexnames]
    
    res_fi.columns = colnames
    res_fi.index = indexnames
    res_sd.columns = colnames
    res_sd.index = indexnames
    
    
    res_fi.to_csv(f'{results_folder}/{comb_name}_RFC_contributions_table.csv')
    res_sd.to_csv(f'{results_folder}/{comb_name}_RFC_contributions_sd_table.csv')
    
    fig, ax = plt.subplots(figsize=(25, 11))
    cmap = sns.cm.rocket_r
    sns.heatmap(res_fi, cmap = cmap, annot=False, fmt='.2f', ax=ax)
    plt.savefig(f'{results_folder}/{comb_name}_FINAL_RFC_contributions')
    plt.close()
    
    sns.clustermap(res_fi, cmap=cmap, xticklabels=True, figsize=(25, 11))
    #sns.color_palette("rocket", as_cmap=True)
    plt.savefig(f'{results_folder}/{comb_name}_RFC_contributions_cluster')
    plt.close()
    #
    # fig, axs = plt.subplots(nrows=len(indexnames), figsize=(10, 50), sharex='all')
    #
    # for i, (mean_row, std_row) in enumerate(zip(res_fi.iterrows(), res_sd.iterrows())):
    #     index = mean_row[0]
    #     means = mean_row[1]
    #     stds = std_row[1]
    #
    #     axs[i].bar(means.index, means, yerr=stds, capsize=5)
    #     # axs[i].set_xlabel(colnames)
    #     axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)
    #     axs[i].set_ylabel(f'{index}')
    #
    # fig.tight_layout()
    # fig.savefig("FINAL_contributions_bar_plots.png")
    # plt.close()
    
    fig, axs = plt.subplots(nrows=23, figsize=(20, 50), sharex=True, sharey=True, gridspec_kw={"bottom": 0.2})
    
    for i, (mean_row, std_row) in enumerate(zip(res_fi.iterrows(), res_sd.iterrows())):
        index = mean_row[0]
        data = mean_row[1]
        errors = std_row[1]
    
        ax = sns.barplot(x=data.index, y=data.values, ax=axs[i], capsize=.2, ci="sd", color="grey")
        ax.errorbar(x=data.index, y=data.values, yerr=errors.values, fmt='none', ecolor='black', capsize=2)
    
        axs[i].set_xlabel("")
        axs[i].set_ylabel(index)
        axs[i].tick_params(axis='x', rotation=90)
    
    axs[0].xaxis.set_ticks_position('top')
    axs[0].xaxis.set_label_position('top')
    
    fig.savefig(f'{results_folder}/{comb_name}_contributions_bar_plots.svg')
    plt.close()

#############################################################################

def extract_contributions(basemodels_file, datafile, comb_name, results_folder, combinations):   
    base_models = unpickle_objects(basemodels_file)
    dataset = unpickle_objects(datafile)
    #measures = ['EIGENVECTOR','BETWEENNESS','CLOSENESS', 'PAGERANK','AVNEIGHBOUR', 'HARMONIC', 'INFORMATION', 'CONSTRAINT', 'ECCENTRICITY',
    #              'SUBGRAPH', 'APPROXCURRBET','CLIQUENO' ]
    #disc = ['DISCRETIZED']
    #base_omics = ['RPPA','RNA', 'DNA' ]
    omics_list = combinations
    
    features_names_dict = {}
    for omic in omics_list:
        features_names_dict[omic] = list(dataset.to_pandas(omic=omic).columns)
    all_features_names = list(dataset.extract(omics_list=omics_list).dataframe.columns)
    targets_names = base_models.keys()
    
    fi_dict = {}
    
    for target_name, target_dict in base_models.items():
        rr = pd.DataFrame(columns=all_features_names)
        idx = 0
        for loop1_number, loop1_dict in target_dict.items():
            for loop2_number, loop2_dict in loop1_dict.items():
                for omic_name, omic_dict in loop2_dict.items():
                    for algo_name, algo_data in omic_dict.items():
                        if not algo_name in ['Ridge', 'KNN', 'SVC']:
                            if algo_name in ['SVM', 'Logistic', 'EN']:
                                algo_data = algo_data[0]
                            print(f'{target_name}/{loop1_number}/{loop2_number}/{omic_name}/{algo_name}')
    
                            sorted_indices = np.argsort(-algo_data)
                            ranks = np.empty_like(sorted_indices)
                            ranks[sorted_indices] = np.arange(len(algo_data))
                            rr.loc[idx, features_names_dict[omic_name]] = ranks
                            idx += 1
    
        rr.loc['average', :] = rr.mean(axis=0)
        fi_dict[target_name] = rr
        for this_key in fi_dict.keys():
                fi_dict[this_key].to_csv(f'{results_folder}/fi_{this_key}_{comb_name}.csv')
        fi_dict = {}
    
#    for this_key in fi_dict.keys():
#        fi_dict[this_key].to_csv(f'{results_folder}/fi_{this_key}_{comb_name}.csv')

#############################################################

def extract_features(datafile, comb_name, results_folder, combinations):
    file_list = glob.glob(f'{results_folder}/fi_*')
    #file_list.remove('FI_plots')
    #dataset = unpickle_objects('Topo_integration_2023-05-01-06-25-23-229169.pkl')
    dataset = unpickle_objects(datafile)
    
    
    omics_list = combinations
    algos_list = ['SVC', 'RFC', 'Logistic', 'EN', 'ET', 'XGB', 'Ada']
    omics_biglist = dataset.omic
    
    all_col_meds = pd.DataFrame(index=dataset.dataframe.columns)
    
    for file in file_list:
        df = pd.read_csv(file)
        drugname = file.split('_')[1]
        for omic in omics_list:
         #   fig, axs = plt.subplots(nrows=8, figsize=(8, 90))
            these_features = omics_biglist[omics_biglist == omic].index
            idxs = [x for x in df.columns if x in these_features]
            this_df = df.loc[:, idxs]
            this_df = this_df.iloc[:-1, :]
            # col_means = this_df.mean()
            col_meds = this_df.median()
            # all_col_meds = pd.concat([all_col_meds, col_meds], axis=1)
            # sorted_cols = col_means.sort_values()
            sorted_cols = col_meds.sort_values()
            sorted_cols = sorted_cols.index[:30]
            sorted_df = this_df[sorted_cols].dropna()
           # ax = axs[0]
            colnames = sorted_df.columns
            colnames = [x.split('_ENS')[0].split('_Cautio')[0].split('_nmiR')[0].split('/isoval')[0].split('/tauro')[0] for x in colnames]
            sorted_df.columns = colnames
          #  sns.boxplot(data=sorted_df + 1, ax=ax, color='white', orient='h')
          #  sns.set_context('paper')
          #  for line in ax.lines:
          #      if line.get_linestyle() == '-':
          #          line.set_color('black')
          #  ax.set_title('All Algorithms')
          #  ax.set_xticklabels(ax.get_xticks(), rotation=90)
          #  ax.set_ylabel('rank')
    
            for i, algo in enumerate(algos_list):
                idxs = list(range(i, this_df.shape[0], 7))
                algo_df = this_df.iloc[idxs, :]
                col_meds = algo_df.median()
                all_col_meds = pd.concat([all_col_meds, col_meds], axis=1)
                sorted_cols = col_meds.sort_values()
                sorted_cols = sorted_cols.index[:30]
                sorted_df = algo_df[sorted_cols].dropna()
           #     ax = axs[i+1]
                colnames = sorted_df.columns
                colnames = [x.split('_ENS')[0].split('_Cautio')[0].split('_nmiR')[0].split('/isoval')[0].split('/tauro')[0] for x in colnames]
                sorted_df.columns = colnames
           #     sns.boxplot(data=sorted_df + 1, ax=ax, color='white', orient='h')
                sorted_df.to_csv(f"{results_folder}/{comb_name}_FI_med_{drugname}_{omic}_{algo}")
           #     sns.set_context('paper')
           #     for line in ax.lines:
              #      if line.get_linestyle() == '-':
              #          line.set_color('black')
              #  ax.set_title(algo)
              #  ax.set_xticklabels(ax.get_xticks(), rotation=90)
              #  ax.set_ylabel('rank')
    
            plt.suptitle(f'Distribution of features ranks: {drugname} / {omic}')
            plt.tight_layout()
            plt.savefig(f'{results_folder}/{comb_name}_FI_med_{drugname}_{omic}.tif', format='tif', dpi=300)
            plt.close()
    
    all_col_meds.to_csv(f"{results_folder}/{comb_name}_all_col_meds.csv")
    
    
    
    
########################################################

