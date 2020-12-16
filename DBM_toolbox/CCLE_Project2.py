# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:44:18 2020

@author: sebastien.delandtshe
"""

import DBM_toolbox as DBM

from sklearn.model_selection import train_test_split
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from sklearn import preprocessing as pp
from sklearn import metrics
from datetime import datetime
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import random
import venn

import numba
#Acceleration
@numba.jit
def f(x):
	return x
@numba.njit
def f(x):
	return x


def rescale_data(df):
	"""Normalization by mapping to the [0 1] interval (each feature independently)
	this is the same as maxScaler? should we leave it?"""
	return (df - df.min()) / (df.max() - df.min())



###############################################################################
###############################################################################
# Import the data
df_prot = pd.read_csv('../data/CCLE_RPPA_20181003.csv')
df_prot = df_prot.set_index('Unnamed: 0')
df_prot = DBM.rescale_data(df_prot)
df_prot = np.log2(df_prot + 1)

df_RNA = pd.read_csv('../data/CCLE_RNASeq_genes_rpkm_20180929.csv')
df_RNA['GeneTrans'] = df_RNA['Description'] + '_' + df_RNA['Name']
df_RNA = df_RNA.set_index(['GeneTrans'])
df_RNA = df_RNA.drop(['Description', 'Name'], axis=1)
df_RNA = df_RNA.transpose()
df_RNA = np.log2(df_RNA + 1)
df_RNA = DBM.eliminate_sparse_data(df_RNA, col_threshold = 0.99, row_threshold = 0.99)

df_miRNA = pd.read_csv('../data/CCLE_miRNA_20181103.csv')
df_miRNA['GeneTrans'] = df_miRNA['Description'] + '_' + df_miRNA['Name']
df_miRNA = df_miRNA.set_index(['GeneTrans'])
df_miRNA = df_miRNA.drop(['Description', 'Name'], axis=1)
df_miRNA = df_miRNA.transpose()
df_miRNA = np.log2(df_miRNA + 1)

df_drug = pd.read_csv('../data/CCLE_NP24.2009_Drug_data_2015.02.24.csv')
df_drug = DBM.reformat_drugs(df_drug)


#GDSC expression file
df_RNA2 = pd.read_csv('../data/Cell_line_RMA_proc_basalExp.txt', sep='\t')


#GDSC drug file
df_drug2 = pd.read_csv('../data/GDSC2_fitted_dose_response_25Feb20.csv', sep=';')



###############################################################################
# Data exploration
n_drug = set(list(df_drug.index.values))
n_prot = set(list(df_prot.index.values))
n_rna = set(list(df_RNA.index.values))
n_mirna = set(list(df_miRNA.index.values))

datasets = {
	'Drug Screen': n_drug, 'RPPA': n_prot, 'RNA': n_rna, 'miRNA': n_mirna}
venn.venn(datasets)
plt.show() #451 samples with all four data types


# Work only on the intersect of the four datasets
tmp = pd.merge(df_drug, df_prot, left_index = True, right_index = True)
tmp2 = pd.merge(tmp, df_RNA, left_index = True, right_index = True)
df = pd.merge(tmp2, df_miRNA, left_index = True, right_index = True)
idx = df.index

dfx_prot = df_prot.loc[idx, :]
dfx_rna = df_RNA.loc[idx, :]
dfx_mirna = df_miRNA.loc[idx, :]
dfx_drugs = df_drug.loc[idx, :]

# visualization before transformations
dfx_prot_sample = dfx_prot.iloc[:, random.sample(range(0, dfx_prot.shape[1]), 100)]
DBM.show_me_the_data(dfx_prot_sample)
dfx_rna_sample = dfx_rna.iloc[:, random.sample(range(0, dfx_rna.shape[1]), 100)]
DBM.show_me_the_data(dfx_rna_sample)
dfx_mirna_sample = dfx_mirna.iloc[:, random.sample(range(0, dfx_mirna.shape[1]), 100)]
DBM.show_me_the_data(dfx_mirna_sample)

dfx_prot.info()
dfx_rna.info()
dfx_mirna.info()

dfx_prot = DBM.remove_constant_data(dfx_prot)
dfx_rna = DBM.remove_constant_data(dfx_rna)
dfx_mirna = DBM.remove_constant_data(dfx_mirna)

dfx_prot = DBM.rescale_data(dfx_prot)
dfx_rna = DBM.rescale_data(dfx_rna)
dfx_mirna = DBM.rescale_data(dfx_mirna)

# DBM.explore_shape(df, plot = True)

# Add the tumor type
tumors_list = ['PROSTATE', 'STOMACH', 'URINARY', 'NERVOUS', 'OVARY', 'HAEMATOPOIETIC',
	'KIDNEY', 'THYROID', 'SKIN', 'SOFT_TISSUE', 'SALIVARY', 'LUNG', 'BONE',
	'PLEURA', 'ENDOMETRIUM', 'BREAST', 'PANCREAS', 'AERODIGESTIVE', 'LARGE_INTESTINE',
	'GANGLIA', 'OESOPHAGUS', 'FIBROBLAST', 'CERVIX', 'LIVER', 'BILIARY', 
	'SMALL_INTESTINE']
df_tumors = pd.DataFrame(index=df.index, columns=tumors_list)
df_tumors_rna = pd.DataFrame(index=df_RNA.index, columns=tumors_list)


df_tissue = pd.Series(index=df.index)
for col, id in enumerate(tumors_list):
	for lin, c in enumerate(df.index):
		if id in c:
			df_tumors.iloc[lin, col] = 1.0
			df_tissue.iloc[lin] = id
		else:
			df_tumors.iloc[lin, col] = 0.0

for col in df_tumors.columns:
	df_tumors[col] = pd.to_numeric(df_tumors[col])
	

df_tissue_rna = pd.Series(index=df_RNA.index)
for col, id in enumerate(tumors_list):
	for lin, c in enumerate(df_RNA.index):
		if id in c:
			df_tumors_rna.iloc[lin, col] = 1.0
			df_tissue_rna.iloc[lin] = id
		else:
			df_tumors_rna.iloc[lin, col] = 0.0

for col in df_tumors.columns:
	df_tumors_rna[col] = pd.to_numeric(df_tumors_rna[col])

df_melanoma = df_tumors_rna[df_tumors_rna['SKIN']==1]
list_melanoma = df_melanoma.index

# df = pd.merge(df, df_tumors, left_index = True, right_index = True)

tumor_types = df_tumors.sum().sort_values(ascending=False)
ax = plt.subplot()
plt.bar(x=tumor_types.index, height=tumor_types)
ax.set_title('Tumor types')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

datatype = ['RPPA', 'RNA', 'miRNA']
for dt, dataset in enumerate([dfx_prot, dfx_rna, dfx_mirna]):
	data_mean = dataset.mean(axis=0)
	data_std = dataset.std(axis=0)
	f2, ax = plt.subplots()
	plt.plot(data_mean, data_std, 'ko')
	plt.title(datatype[dt])


list_drugs = list(set([x.split('_')[0] for x in dfx_drugs.columns]))
dfx_drugs = DBM.impute_missing_data(dfx_drugs)
dfx_drugs = DBM.rescale_data(dfx_drugs)

# f3, ax = plt.subplots(5,5, sharex = True, sharey=True)
# ax = ax.ravel()
# for idx, drug in enumerate(list_drugs):
# 	cols = [col for col in dfx_drugs.columns if drug in col]
# 	cols2 = [col for col in cols if 'IC50' in col]
# 	cols3 = [col for col in cols if 'ActArea' in col]
# 	dfx_drugs[drug + '_response'], eq = DBM.project_distance(dfx_drugs[cols2].values, dfx_drugs[cols3].values)
# 	sns.scatterplot(x= cols2[0], y=cols3[0], data=dfx_drugs, 
# 				 hue=drug + '_response', palette='ch:2.5,-.8,dark=.3', ax=ax[idx], legend=False)
# 	ax[idx].plot([0, 1], [eq[1], eq[0]+eq[1]])
# 	ax[idx].set_title(drug)
# 	ax[idx].set_xlabel('IC50')
# 	ax[idx].set_ylabel('ActArea')

common = ['Erlotinib', 'PHA-665752', 'Paclitaxel', 'Sorafenib', 'TAE684', 'Crizotinib',
		   'AZD-0530', 'Lapatinib', 'Nilotinib', '17-AAG', 'PLX4720', 'PD-0332991',
		    'PD-0325901', 'AZD6244', 'Nutlin-3']


responses = [col for col in dfx_drugs.columns if ('ActArea' in col)]
dfx_drugs = dfx_drugs[responses]
dfx_drugs = dfx_drugs.iloc[:, [1, 3, 4, 5, 6, 11, 15, 16, 17, 18, 19, 21, 23]]
dfx_drugs = DBM.rescale_data(dfx_drugs)
dfx_drugs.hist(figsize=(15, 20), bins=50, xlabelsize=8, ylabelsize=8)

res = DBM.get_PCA(dfx_prot, n_components = min(dfx_prot.shape))
dfx_prot_PCs = res[0]
res = DBM.get_PCA(dfx_rna, n_components = min(dfx_rna.shape))
dfx_rna_PCs = res[0]
res = DBM.get_PCA(dfx_mirna, n_components = min(dfx_mirna.shape))
dfx_mirna_PCs = res[0]

dfx_prot_ICs = DBM.get_ICA(dfx_prot, n_components = 20)
dfx_rna_ICs = DBM.get_ICA(dfx_rna, n_components = 100)
dfx_mirna_ICs = DBM.get_ICA(dfx_mirna, n_components = 20)

dfx_prot_RPCs = DBM.get_RPCA(dfx_prot, n_components = 20)
dfx_rna_RPCs = DBM.get_RPCA(dfx_rna, n_components = 100)
dfx_mirna_RPCs = DBM.get_RPCA(dfx_mirna, n_components = 20)

tmp = pd.merge(dfx_prot, dfx_rna, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_mirna, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_prot_PCs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_rna_PCs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_mirna_PCs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_prot_ICs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_rna_ICs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_mirna_ICs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_prot_RPCs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_rna_RPCs, left_index = True, right_index = True)
tmp = pd.merge(tmp, dfx_mirna_RPCs, left_index = True, right_index = True)
df_complete = pd.merge(tmp, df_tumors, left_index = True, right_index = True)

df_all = [dfx_prot, dfx_rna, dfx_mirna,
		   dfx_prot_PCs, dfx_rna_PCs, dfx_mirna_PCs, 
		   dfx_prot_ICs, dfx_rna_ICs, dfx_mirna_ICs, 
		   dfx_prot_RPCs, dfx_rna_RPCs, dfx_mirna_RPCs, 
		   df_tumors, df_complete]

list_datasets = ['prot', 'rna', 'mirna', 
				 'protPC', 'rnaPC', 'mirnaPC', 
				 'protIC', 'rnaIC', 'mirnaIC', 
				 'protRP', 'rnaRP', 'mirnaRP', 
				 'tumortype', 'complete']

df_all = [dfx_rna]
list_datasets = ['rna']



#%% modeling

filename = 'debug_'+((datetime.now()).strftime("%Y%m%d%H%M%S"))+'.log'

res_indiv_perfs = []
res_indiv_models = []
res_stack_perfs = []
res_indiv_importances = []
res_stack_importances = []

T = [0.333, 1-0.333]

targets = dfx_drugs.iloc[:,4].to_frame()
targets = dfx_drugs

final_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, learning_rate=0.1, n_estimators=200, max_depth=4)

import DBM_toolbox as DBM
for i, target in enumerate(targets.columns):
	
	targets_bn = DBM.get_drug_response(targets, thresholdR = targets.quantile(T[0]), thresholdS = targets.quantile(T[1]), axis = 'columns')
	retained_idx = targets_bn.index[targets_bn.loc[:,target]!=0]
	y = targets_bn.loc[retained_idx, target]
	y_quant = targets.loc[retained_idx, target]
	y[y==-1]=0
	
	for idx_dataset, df in enumerate(df_all):
		this_dataset = list_datasets[idx_dataset]
		if idx_dataset == 1:
			df2 = DBM.remove_constant_data(df, method = 'variance', threshold = Var) # % most variable data
		else:
			df2 = df
	
		X_red = df2.loc[retained_idx, :]
		X, bestFeatures, featureScores = DBM.select_top_features(
			X_red, y, problem = 'classification', n_best = 40, method = 'importance', plots=False) #remove non-informative features
		
		resopt, summary, list_models, list_opt_models, list_trained_models = DBM.get_optimized_models(X, y, n_trials = 50)
		Lim = np.int(X.shape[0]*(4/5))
		X_train = X.iloc[0:Lim, :]
		X_test = X.iloc[Lim:, :]
		y_train = y.iloc[0:Lim]
		y_test = y.iloc[Lim:]
		fitted_stack, perf_test, perf_train, y_pred_test, y_proba = DBM.stack_models(list_opt_models, final_model, X, y, X_test, y_test, folds = 5)
		
# 		for j, thismodel in list_trained_models:
# 			try:
# 				this_coef = 
# 		
# 		pd.Series(abs(thismodel.coef_[0]), index=X.columns).nlargest(10).plot(kind='barh')
		
		
		res_indiv_perfs.append(resopt) #summary.iloc[0:-1, 0]
		res_indiv_models.append(list_trained_models)
		res_stack_perfs.append(perf_train)
# 		res_indiv_importances.append()
		res_stack_importances.append(fitted_stack.feature_importances_)

	
#%%	
	
	
	
	
	
	
	
	
	
	
	
	
# 	idx_dataset, df in enumerate(df_all):
	
	targets = dfx_drugs.iloc[:,4]
	
	print('Target: %s, data: %s, top %.2f vs bottom %.2f when keeping %.2f of the data' % (target, this_dataset, Prop, Prop, 1-Var))
	T = [Prop, 1-Prop]
	
	targets_bn = DBM.get_drug_response(targets, thresholdR = targets.quantile(T[0]), thresholdS = targets.quantile(T[1]), axis = 'columns')
	dataset_idx = targets_bn.index[targets_bn.loc[:,target]!=0]
	if idx_dataset == 1:
		df2 = DBM.remove_constant_data(df, method = 'variance', threshold = Var) # % most variable data
	else:
		df2 = df

# 	final_results = []
	reps = 5 #outer x-val
	folds = 5 #inner x-val

	Prop = 0.333
	Var = 0.2
	if this_dataset in ['rna', 'complete']:
		Var = 0.80
	
	###for testing only on 1 column:
	targets = targets.to_frame()
	
# 	fig, ax = plt.subplots()
	feature_variance = np.std(df)
	feature_mean = np.mean(df)
# 	plt.scatter(feature_mean, feature_variance)
	
	for i, target in enumerate(targets.columns):
		# create figure x by x
		print('Target: %s, data: %s, top %.2f vs bottom %.2f when keeping %.2f of the data' % (target, this_dataset, Prop, Prop, 1-Var))
		
		targets_bn = DBM.get_drug_response(targets, thresholdR = targets.quantile(T[0]), thresholdS = targets.quantile(T[1]), axis = 'columns')
		dataset_idx = targets_bn.index[targets_bn.loc[:,target]!=0]
		if idx_dataset == 1:
			df2 = DBM.remove_constant_data(df, method = 'variance', threshold = Var) # % most variable data
		else:
			df2 = df
		
		fid = open(filename, 'a')
		fid.write(this_dataset + '\n')
		fid.write(target + '\n')
		fid.write('dataset size:' + '\n')
		fid.write(str(df.shape) + '\n')
		fid.write('dataset size only high variance:' + '\n')
		fid.write(str(df2.shape) + '\n')
		fid.close()
		
		X_red = df2.loc[dataset_idx, :]
		y = targets_bn.loc[dataset_idx, target]
		y_quant = targets.loc[dataset_idx, target]
		y[y==-1]=0
		
		X, bestFeatures, featureScores = DBM.select_top_features(
			X_red, y, problem = 'classification', n_best = 40, method = 'importance', plots=False) #remove non-informative features
		all_feat.append(featureScores)
		#this adds boolean and polynomial combinations of the best features to the dataset
	# 	Pol = DBM.get_polynomials(X_red_info[bestFeatures], degree = 2)
	# 	Bool = DBM.get_boolean(X_red_info[bestFeatures])
	# 	
	# 	X = pd.merge(X_red_info, Pol, left_index=True, right_index=True)
	# 	X = pd.merge(X, Bool, left_index=True, right_index=True)
	# 	
	# 	fid = open(filename, 'a')
	# 	fid.write('dataset size only informative:' + '\n')
	# 	fid.write(str(X_red_info.shape) + '\n')
	# 	fid.write('dataset size after poly-boolean engineering:' + '\n')
	# 	fid.write(str(X.shape) + '\n')
	# 	fid.close()
		resopt, summary, list_models, list_opt_models, list_trained_models = DBM.get_optimized_models(X, y, n_trials = 50)
		Lim = np.int(X.shape[0]*(4/5))
		X_train = X.iloc[0:Lim, :]
		X_test = X.iloc[Lim:, :]
		y_train = y.iloc[0:Lim]
		y_test = y.iloc[Lim:]
		fitted_stack, perf_test, perf_train, y_pred_test, y_proba = DBM.stack_models(list_opt_models, 
																			   final_model, X, y, X_test, y_test, folds)
		
		results.append(summary.iloc[0:-1, 0])
		all_opt_models.append(list_opt_models)
		all_stacks.append(fitted_stack)
		all_importances.append(fitted_stack.feature_importances_)
		stack_results.append(perf_train)
		
	#%%
thismodel = list_opt_models[1]
thismodel.fit(X, y)

pd.Series(abs(thismodel.coef_[0]), index=X.columns).nlargest(10).plot(kind='barh')


# features_names = ['input1', 'input2']
# svm = svm.SVC(kernel='linear')
# svm.fit(X, Y)
# f_importances(svm.coef_, features_names)
# 	
# 	
	#%%
		
		skf_outer = StratifiedKFold(n_splits=reps, shuffle= True, random_state=42)
		
		outer_scores_train = []
		outer_scores_test = []
		outer_subsets = []
		foldwise_predictions = [0]*len(X)
		
		n_outerfold = 0
		for train_index_outer, test_index_outer in skf_outer.split(X, y):
			n_outerfold = n_outerfold +1
			X_train, X_test = X.iloc[train_index_outer, :], X.iloc[test_index_outer, :]
			y_train, y_test = y.iloc[train_index_outer], y.iloc[test_index_outer]
			
			resopt, summary, list_models, list_opt_models = DBM.get_optimized_models(X_train, y_train, n_trials = 30)
			
		results.append(summary)
		all_feat.append(list_opt_models)
		
		
		
#%%
	# 		thisNames = list_models
			list_opt_models = []
			for item in resopt:
				list_opt_models.append(item[1])
			isbest = summary['auc'] == max(summary['auc'])
			best_idx = next((i for i, j in enumerate(isbest) if j), None)
			best_score = 0
			best_subset = [list_opt_models[best_idx]]
			remaining = list_opt_models.copy()
			remaining.remove(list_opt_models[best_idx])
			dmp1, best_score, dmp2 , dmp3 , dmp4 = DBM.stack_models(best_subset, final_model, X_train, y_train, X_test, y_test, folds)
			
			search = True
			while search:
				search = False
				for trial in remaining:
					this_best_subset = best_subset.copy()
					if trial != 0:
						this_best_subset.append(trial)
						print('Target: %s, fold %d/%d, trying to add %s to the stack...' % (target, n_outerfold, reps, trial), end='')
						fitted_stack, perf_test, perf_train, y_pred_test, y_proba = DBM.stack_models(this_best_subset, final_model, X_train, y_train, X_test, y_test, folds)
						print(' training: %.3f, test: %.3f' % (perf_train, perf_test))
						
						if perf_test > (best_score*(1+min_improvement)):
							search = True
							best_subset = this_best_subset
							remaining = list_opt_models.copy()
							for toremove in best_subset:
								remaining.remove(toremove)
							best_score = perf_test
							print('*** A better stack was found (test = %.3f) by adding %s ***' % (perf_test, trial))
							print('%s Optimal stack so far... %s' % ('*'*20, '*'*20))
							print(best_subset)
					else:
						remaining.remove(trial)
				
			print('%s Optimal stack was found with score %.3f %s ' % ('*'*20,best_score,'*'*200))
			print(best_subset)
			
			fid = open(filename, 'a')
			fid.write('optimal stack: AUC')
			fid.write(str(best_subset))
			fid.write(str(best_score) + '\n')
			
			outer_scores_train.append(perf_train)
			outer_scores_test.append(best_score)
			outer_subsets.append(best_subset)
			
			for c, idx in enumerate(test_index_outer):
				foldwise_predictions[idx] = y_proba[c,1]
			#%%
		
		final_subset=[]
		outer_list = list(set([item for list2 in outer_subsets for item in list2]))
		for item in outer_list:
			if item not in outer_subsets:
				final_subset.append(item)
		print('*'*50 + ' final subset: ' + '*'*200)
		print(final_subset)
		estim_score_train = np.mean(outer_scores_train)
		estim_std_train = np.std(outer_scores_train)
		estim_score_test = np.mean(outer_scores_test)
		estim_std_test = np.std(outer_scores_test)
		fig, ax = plt.subplots(1,2, figsize=(15,15))
		title = target + ', ' + list_datasets[idx_dataset] + ', AUC during building = ' + ('%.3f' % estim_score_test) + ' +- ' + ('%.3f' % estim_std_test)
		fig.suptitle(title)
		_, mean_scores, std_scores = DBM.plot_roc_validation(X, y, final_model, final_subset, reps=2, folds=10, ax=ax[0])
		final_results.append([target, this_dataset, mean_scores, std_scores, y_quant, foldwise_predictions, final_subset, estim_score_test, estim_std_test])
		
		ax[1].scatter(y_quant, foldwise_predictions)
		ax[1].plot([-0.02,1.02],[0.5,0.5],'r--')
		
		item = final_results[-1]
		y = (y_quant > y_quant.median()).astype(int)
		rs = DBM.return_PPV_NPV(y, foldwise_predictions, df_tissue, tumors_list, target = target)
		fid = open(filename, 'a')
		fid.write(target+'\n')
		fid.write(rs.to_string() + '\n')
		
		for comp in final_subset:
# 			fid.write(comp.__class__.__name__ + '\n')
			fid.write(comp + '\n')
		fid.write('\n'+'*'*20+'\n')
		
		results_all_df.append([final_results, rs])
	
	fid.close()
	
	
	
	
	
	

#%%

for idx_dataset, df in enumerate(df_all):
	this_dataset = list_datasets[idx_dataset]
	targets = dfx_drugs.iloc[:, 2]
	
	
	resopt, summary, list_models = DBM.get_optimized_models(X_train, y_train, n_trials = 30)
	
