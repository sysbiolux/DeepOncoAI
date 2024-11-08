###################################################################################################################
# DEEP-ONCO-AI is an analysis pipeline that trains a metaclassifier on the predictions of machine-learning algos  #
# in order to predict the chemosensitivity of cell lines in the CCLE dataset. The first-level algos are trained   #
# independently for each drug and omic type. Most important features of the most contributing algos are retrieved #
# and used to form explainable classifiers. The contributions of the different omics is systematically examined.  #
###################################################################################################################

# Authors: Sebastien De Landtsheer: sebdelandtsheer@gmail.com
#          Prof Thomas Sauter, University of Luxembourg

# This version: February 2023

####################
### HOUSEKEEPING ###
####################

import logging
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from DBM_toolbox.data_manipulation import data_utils, dataset_class

from config import Config  # many operations are conducted from the Config class, as it has access to the config file

logging.basicConfig(
    filename="config_test_topo_integration_2.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

rng = np.random.default_rng(42)

outer_folds = 5
inner_folds = 5

config = Config("testall/config_test_topo_integrated.yaml")  # here is the path to the config file to be used in the analysis

###################################
### READING AND PROCESSING DATA ###
###################################

# logging.info("Reading data")
# data, ActAreas, ic50s, dose_responses = config.read_data(join_type = 'inner') #join_type = 'inner'

# #### Uncomment if running for individual cancers
# lung_samples = [x for x in data.dataframe.index if 'LARGE_INTESTINE' in x]
# not_lung_samples = [x for x in data.dataframe.index if 'LARGE_INTESTINE' not in x]
# actual_data, not_necessary = data.split(train_index=lung_samples, test_index=not_lung_samples)
# filtered_data, filters = config.filter_data(actual_data)
# #####

# # logging.info("Creating visualizations")
# # config.visualize_dataset(data, ActAreas, IC50s, dose_responses, mode="pre")

# logging.info("Filtering data")

# # ### Uncomment if running for different cancer types
# filtered_data, filters = config.filter_data(data)
# # # #####

# logging.info("Selecting subsets for feature engineering")
# selected_subset = config.select_subsets(filtered_data)

# logging.info("Engineering features")
# engineered_features = config.engineer_features(filtered_data)

# logging.info("Merging engineered features")
# engineered_data = filtered_data.merge_with(engineered_features)

# logging.info("Quantizing targets")
# quantized_data = config.quantize(engineered_data, target_omic="DRUGS", ic50s=ic50s)

# final_data = quantized_data.normalize().optimize_formats()
# config.save(to_save=final_data, name="Topo_integration")

##########################################################################


final_data = data_utils.unpickle_objects('Topo_integration_2023-05-06-19-07-06-236917.pkl')
refiltered_data = final_data
for ref_omic in ['EIGENVECTOR',
                 'BETWEENNESS',
                 'CLOSENESS',
                 'PAGERANK',
                 'AVNEIGHBOUR',
                 'HARMONIC',
                 'INFORMATION',
                 'CONSTRAINT',
                 'ECCENTRICITY',
                 'SUBGRAPH',
                 'APPROXCURRBET',
                 'CLIQUENO',
                 'SQUARECLUSTERING',
                 'DEGREECENT',
                 'DISCRETIZED'
                 ]:
    print(ref_omic)
    refiltered_data = refiltered_data.filter_att(target_omic = 'RNA', reference_omic= ref_omic, separator='_')
    
config.save(to_save=refiltered_data, name="Topo_PARTIAL")
# missing_data = final_data.dataframe.loc[:, final_data.dataframe.isnull().any(axis=0)]

# ######

# logging.info("Getting optimized models")

# trained_models = config.get_models(dataset=final_data, method="standard")
# config.save(to_save=trained_models, name="f_test_toy_models")



# ########################## to load previous data

#final_data = data_utils.unpickle_objects('Topo_integration_2023-05-01-06-25-23-229169.pkl')
#trained_models = config.get_models(dataset=final_data, method="standard")
#config.save(to_save=trained_models, name="Topo_integrated_models")

#
# ff = pd.DataFrame.from_dict(trained_models)
# ff2 = ff.T
# ff2.columns = ['RNA']
# ff3 = ff2.T
# final_ff = pd.DataFrame.to_dict(ff3)
#
# trained_models = final_ff

# final_results = dict()
# feature_importances = dict()
# base_models = dict()
# fprs = dict()

# tprs = dict()
# roc_aucs = dict()

# data = [x for x in final_data.dataframe.columns if 'ActArea' not in x]
# l = len(data)
# targets = [x for x in final_data.dataframe.columns if 'ActArea' in x]
# omics = final_data.omic[data]
# measures = ['EIGENVECTOR','BETWEENNESS','CLOSENESS', 'PAGERANK','AVNEIGHBOUR', ]
# combinations = ['RPPA','RNA', 'DNA' ]

# for target in targets:
#     base_models[target] = dict()
#     omics[target] = 'DRUGS'
#     this_df = pd.concat([final_data.dataframe[data], final_data.dataframe[target]], axis=1)
#     this_df = dataset_class.Dataset(this_df, omic=omics, database='x')
#     this_df = this_df.to_binary(target=target)

#     n_samples = len(this_df.dataframe.index)
#     split_size = np.floor(n_samples / outer_folds)
#     outer_mixed = rng.choice(this_df.dataframe.index, size=n_samples, replace=False)

#     outer_train_idx = dict()
#     outer_test_idx = dict()
#     inner_train_idx = dict()
#     inner_test_idx = dict()
#     feature_importances[target] = dict()
#     final_results[target] = pd.DataFrame(this_df.dataframe[target].copy())  # truth

#     for outer_loop in range(outer_folds):
#         feature_importances[target][outer_loop] = dict()
#         if outer_loop == outer_folds - 1:  # if it is the last split, put all remaining samples in
#             outer_test_idx[outer_loop] = outer_mixed[int(outer_loop * split_size): -1]
#         else:
#             outer_test_idx[outer_loop] = outer_mixed[int(outer_loop * split_size): int((outer_loop + 1) * split_size)]
#         outer_train_idx[outer_loop] = [x for x in outer_mixed if x not in outer_test_idx[outer_loop]]

#         rest_dataset, valid_dataset = this_df.split(train_index=outer_train_idx[outer_loop],
#                                                        test_index=outer_test_idx[outer_loop])

#         n_inner_samples = len(rest_dataset.dataframe.index)
#         split_inner_size = np.floor(n_inner_samples / inner_folds)
#         inner_mixed = rng.choice(rest_dataset.dataframe.index, size=n_inner_samples, replace=False)

#         inner_train_idx[outer_loop] = dict()
#         inner_test_idx[outer_loop] = dict()
#         base_models[target][outer_loop] = dict()

#         first_level_preds = pd.DataFrame(this_df.dataframe[target].copy())  # truth

#         for inner_loop in range(inner_folds):
#             print(f"target: {target}, out: {outer_loop+1}/{outer_folds}, in: {inner_loop+1}/{inner_folds}")
#             if inner_loop == inner_folds - 1:
#                 inner_test_idx[outer_loop][inner_loop] = inner_mixed[int(inner_loop * split_inner_size): -1]
#             else:
#                 inner_test_idx[outer_loop][inner_loop] = inner_mixed[int(inner_loop * split_inner_size): int((inner_loop + 1) * split_inner_size)]
#             inner_train_idx[outer_loop][inner_loop] = [x for x in inner_mixed if x not in inner_test_idx[outer_loop][inner_loop]]

#             train_dataset, test_dataset = rest_dataset.split(train_index=inner_train_idx[outer_loop][inner_loop], test_index=inner_test_idx[outer_loop][inner_loop])

#             base_models[target][outer_loop][inner_loop] = dict()

#             for omic in trained_models[target].keys():  # for each omic type
#                 if omic == 'DISCRETIZED' or omic == 'EIGENVECTOR':
#                     print(f'omic: {omic}')
#                     train_features = train_dataset.to_pandas(omic=omic)
#                     train_labels = train_dataset.to_pandas(omic='DRUGS').values.ravel()
#                     test_features = test_dataset.to_pandas(omic=omic)
#                     test_labels = test_dataset.to_pandas(omic='DRUGS').values.ravel()

#                     base_models[target][outer_loop][inner_loop][omic] = dict()

#                     for algo in trained_models[target][omic].keys():  # for each algo
#                         print(f'algo: {algo}')
#                         this_model = trained_models[target][omic][algo]['estimator']
#                         this_model.fit(train_features, train_labels)
#                         try:
#                             this_predictions = this_model.predict_proba(test_features)
#                             this_predictions = this_predictions[:, 1]
#                         except:
#                             this_predictions = this_model.predict(test_features)
#                         first_level_preds.loc[inner_test_idx[outer_loop][inner_loop], 'pred_' + omic + '_' + algo] = this_predictions
#                         try:
#                             base_models[target][outer_loop][inner_loop][omic][algo] = this_model.feature_importances_
#                         except:
#                             try:
#                                 base_models[target][outer_loop][inner_loop][omic][algo] = this_model.coef_
#                             except:
#                                 base_models[target][outer_loop][inner_loop][omic][algo] = 'None'

#         for omic in trained_models[target].keys():  # for each omic type
#             if omic == 'DISCRETIZED' or omic == 'EIGENVECTOR':
#                 print(f'omic: {omic}')
#                 train_features = rest_dataset.to_pandas(omic=omic)
#                 train_labels = rest_dataset.to_pandas(omic='DRUGS').values.ravel()
#                 test_features = valid_dataset.to_pandas(omic=omic)
#                 test_labels = valid_dataset.to_pandas(omic='DRUGS').values.ravel()
#                 for algo in trained_models[target][omic].keys():  # for each algo
#                     print(f'algo: {algo}')
#                     this_model = trained_models[target][omic][algo]['estimator']
#                     this_model.fit(train_features, train_labels)
#                     try:
#                         this_predictions = this_model.predict_proba(test_features)
#                         this_predictions = this_predictions[:, 1]
#                     except:
#                         this_predictions = this_model.predict(test_features)
#                     first_level_preds.loc[outer_test_idx[outer_loop], 'pred_' + omic + '_' + algo] = this_predictions

#         cols = first_level_preds.columns
#         train_features = first_level_preds.loc[outer_train_idx[outer_loop], cols[1:]]
#         train_labels = first_level_preds.loc[outer_train_idx[outer_loop], cols[0]]
#         test_features = first_level_preds.loc[outer_test_idx[outer_loop], cols[1:]]
#         test_labels = first_level_preds.loc[outer_test_idx[outer_loop], cols[0]]
#         train_features, train_labels = data_utils.merge_and_clean(train_features, train_labels)

#         for algo in trained_models[target][omic].keys():
#             if algo == 'RFC':
#                 this_model = trained_models[target][omic][algo]['estimator']
#                 this_model.fit(train_features, train_labels)
#                 try:
#                     this_predictions = this_model.predict_proba(test_features)
#                     this_predictions = this_predictions[:, 1]
#                 except:
#                     this_predictions = this_model.predict(test_features)
#                 try:
#                     f_imp = this_model.feature_importances_
#                 except:
#                     f_imp = np.nan
#                 feature_importances[target][outer_loop][algo] = pd.DataFrame(f_imp, index=this_model.feature_names_in_, columns=[algo])
#                 final_results[target].loc[outer_test_idx[outer_loop], 'pred2_' + algo] = this_predictions

#     omics = omics.drop(target)

# config.save(to_save=final_results, name="f_test_topo2_final_results")
# config.save(to_save=feature_importances, name="f_test_topo2_features")
# config.save(to_save=base_models, name="f_test_topo2_base_models")

