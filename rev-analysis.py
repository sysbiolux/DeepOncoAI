# analyse 10x nested x-val data by cell type:


import numpy as np
import pandas as pd
from config import Config
from matplotlib import pyplot as plt
from DBM_toolbox.data_manipulation.data_utils import pickle_objects, unpickle_objects
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import glob

############## 0.25 vs 0.33 vs 0.475

final_025 = unpickle_objects('REV_025_results_2024-09-16-14-11-19-178278.pkl')
final_033 = unpickle_objects('REV_033_results_')
final_0475 = unpickle_objects('REV_0475_results_')

final_033LR = unpickle_objects('REV_033LR_results_')

targets = final_025.keys()

for target in targets:
    res025 = final_025[target]
    y_true = res025[target]
    y_score = res025['pred2_RFC']
    y_score = y_score.fillna(0.5)
    res025['accurate'] = np.abs(y_true - y_score) < 0.5
    fpr025, tpr025, _ = roc_curve(y_true, y_score)
    roc_auc025 = auc(fpr025, tpr025)

    res033 = final_033[target]
    y_true = res033[target]
    y_score = res033['pred2_RFC']
    y_score = y_score.fillna(0.5)
    res033['accurate'] = np.abs(y_true - y_score) < 0.5
    fpr033, tpr033, _ = roc_curve(y_true, y_score)
    roc_auc033 = auc(fpr033, tpr033)

    res0475 = final_0475[target]
    y_true = res0475[target]
    y_score = res0475['pred2_RFC']
    y_score = y_score.fillna(0.5)
    res0475['accurate'] = np.abs(y_true - y_score) < 0.5
    fpr0475, tpr0475, _ = roc_curve(y_true, y_score)
    roc_auc0475 = auc(fpr0475, tpr0475)

    plt.subplots(figsize=(8, 8))
    plt.plot(fpr025, tpr025, linewidth=3, color='green' , label=f"AUC[0.25]: {round(roc_auc025, 3)}")
    plt.plot(fpr033, tpr033, linewidth=3, color='blue', label=f"AUC[0.33]: {round(roc_auc033, 3)}")
    plt.plot(fpr0475, tpr0475, linewidth=3, color='red', label=f"AUC[0.475]: {round(roc_auc0475, 3)}")
    plt.savefig(f'REV_splits_{target}_ROC')

for target in targets:
    res033 = final_033[target]
    y_true = res033[target]
    y_score = res033['pred2_RFC']
    y_score = y_score.fillna(0.5)
    res033['accurate'] = np.abs(y_true - y_score) < 0.5
    fpr033, tpr033, _ = roc_curve(y_true, y_score)
    roc_auc033 = auc(fpr033, tpr033)

    res033LR = final_033LR[target]
    y_true = res033LR[target]
    y_score = res033LR['pred2_RFC']
    y_score = y_score.fillna(0.5)
    res033LR['accurate'] = np.abs(y_true - y_score) < 0.5
    fpr033LR, tpr033LR, _ = roc_curve(y_true, y_score)
    roc_auc033LR = auc(fpr033LR, tpr033LR)

    plt.subplots(figsize=(8, 8))
    plt.plot(fpr033, tpr033, linewidth=3, color='blue', label=f"AUC[RF]: {round(roc_auc033, 3)}")
    plt.plot(fpr033LR, tpr033LR, linewidth=3, color='red', label=f"AUC[LR]: {round(roc_auc033LR, 3)}")
    plt.savefig(f'REV_integrate_{target}_ROC')






#####################################################################
##################| ROC Curves





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
        plt.plot(fpr[global_model], tpr[global_model], linewidth=3, label=f"AUC: {round(roc_auc[global_model], 3)}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate", fontsize=20, **hfont)
    plt.ylabel("True Positive Rate", fontsize=20, **hfont)
    plt.title(f"{target_name.split('_')[0]}",fontsize=20, **hfont)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(f'Thresholding_25_breast_{target_name}')

    plt.close()

    fprs[target_name] = fpr
    tprs[target_name] = tpr
    roc_aucs[target_name] = roc_auc

roc_aucs_df = pd.DataFrame.from_dict(roc_aucs).T
roc_aucs_df.to_csv('ROC_AUCS_All_meas_all_cancers_RNA.csv')

###################################################################
### clustergrams

fi = unpickle_objects('f_test_toy_features_2023-03-24-13-55-24-586681.pkl')

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


res_fi.to_csv('FINAL_RFC_contributions_table.csv')
res_sd.to_csv('FINAL_RFC_contributions_sd_table.csv')

fig, ax = plt.subplots(figsize=(25, 11))
cmap = sns.cm.rocket_r
sns.heatmap(res_fi, cmap = cmap, annot=False, fmt='.2f', ax=ax)
plt.savefig('FINAL_RFC_contributions')
plt.close()

sns.clustermap(res_fi, cmap=cmap, xticklabels=True, figsize=(25, 11))
#sns.color_palette("rocket", as_cmap=True)
plt.savefig('FINAL_RFC_contributions_cluster')
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

fig.savefig("FINAL_contributions_bar_plots.png")
plt.close()

#############################################################################

base_models = unpickle_objects('f_test_toy_base_models_2023-03-25-12-32-47-823813.pkl')
dataset = unpickle_objects('f_test_topo_1_2023-03-25-09-22-01-645884.pkl')
omics_list = ['RPPA', 'RNA', 'MIRNA', 'META', 'DNA', 'PATHWAYS', 'TYPE']
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

#############################################################

file_list = glob.glob('fi_*')
dataset = unpickle_objects('FINAL_preprocessed_data_2023-02-16-10-30-39-935233.pkl')
omics_list = ['RPPA', 'RNA', 'MIRNA', 'META', 'DNA', 'PATHWAYS', 'TYPE', 'EIGENVECTOR', 'BETWEENNESS','PAGERANK', 'CLOSENESS', 'AVNEIGHBOUR']
algos_list = ['SVC', 'RFC', 'Logistic', 'EN', 'ET', 'XGB', 'Ada']
omics_biglist = dataset.omic

all_col_meds = pd.DataFrame(index=dataset.dataframe.columns)

for file in file_list:
    df = pd.read_csv(file)
    drugname = file.split('_')[1]
    for omic in omics_list:
        fig, axs = plt.subplots(nrows=8, figsize=(8, 90))
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
        ax = axs[0]
        colnames = sorted_df.columns
        colnames = [x.split('_ENS')[0].split('_Cautio')[0].split('_nmiR')[0].split('/isoval')[0].split('/tauro')[0] for x in colnames]
        sorted_df.columns = colnames
        sns.boxplot(data=sorted_df + 1, ax=ax, color='white', orient='h')
        sns.set_context('paper')
        for line in ax.lines:
            if line.get_linestyle() == '-':
                line.set_color('black')
        ax.set_title('All Algorithms')
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.set_ylabel('rank')

        for i, algo in enumerate(algos_list):
            idxs = list(range(i, this_df.shape[0], 7))
            algo_df = this_df.iloc[idxs, :]
            col_meds = algo_df.median()
            all_col_meds = pd.concat([all_col_meds, col_meds], axis=1)
            sorted_cols = col_meds.sort_values()
            sorted_cols = sorted_cols.index[:30]
            sorted_df = algo_df[sorted_cols].dropna()
            ax = axs[i+1]
            colnames = sorted_df.columns
            colnames = [x.split('_ENS')[0].split('_Cautio')[0].split('_nmiR')[0].split('/isoval')[0].split('/tauro')[0] for x in colnames]
            sorted_df.columns = colnames
            sns.boxplot(data=sorted_df + 1, ax=ax, color='white', orient='h')
            sorted_df.to_csv(f"FINAL_FI_med_{drugname}_{omic}_{algo}")
            sns.set_context('paper')
            for line in ax.lines:
                if line.get_linestyle() == '-':
                    line.set_color('black')
            ax.set_title(algo)
            ax.set_xticklabels(ax.get_xticks(), rotation=90)
            ax.set_ylabel('rank')

        plt.suptitle(f'Distribution of features ranks: {drugname} / {omic}')
        plt.tight_layout()
        plt.savefig(f'FINAL_FI_med_{drugname}_{omic}.tif', format='tif', dpi=300)
        plt.close()

all_col_meds.to_csv("all_col_meds.csv")

##########################################################################################

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Results_all_omics_for_plot.csv')

# Create a Seaborn scatterplot
sns.set(style="whitegrid")
scatterplot = sns.scatterplot(x='Total', y='RNA only', data=df, s=25, color='black')

# Add names next to the points
for i in range(len(df)):
    scatterplot.text(df['Total'][i], df['RNA only'][i], df['Drug'][i], ha='left', va='bottom')

plt.xlim(0.45, 1)
plt.ylim(0.45, 1)

# Add a diagonal dashed line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Customize the plot
plt.xlabel('AUROC transcriptomics', fontsize=16)
plt.ylabel('AUROC multiomics', fontsize=16)
plt.title('Comparing performances of multiomics')
# plt.legend(title='Drug', loc='upper left')

# Show the plot
plt.show()