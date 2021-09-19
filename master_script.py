####################
### HOUSEKEEPING ###
####################

import logging
# import numba #does not work?
# @numba.jit
logging.basicConfig(filename='run.log', level=logging.INFO, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
from config import Config
# from DBM_toolbox.data_manipulation import dataset_class
config = Config()

###################################
### READING AND PROCESSING DATA ###
###################################

logging.info("Reading data")
data, IC50s = config.read_data()

# logging.info("Creating visualizations")
# config.visualize_dataset(data, mode='pre')

logging.info("Filtering data")
filtered_data, filters = config.filter_data(data)

print(filtered_data.dataframe.shape)
for omic in list(set(filtered_data.omic)):
    print(f"{omic}: {filtered_data.omic[filtered_data.omic == omic].shape[0]}")

logging.info("Selecting subsets for feature engineering")
selected_subset = config.select_subsets(filtered_data)

logging.info("Engineering features")
if selected_subset is not None:
    engineered_features = config.engineer_features(selected_subset)
    logging.info("Merging engineered features")
    engineered_data = filtered_data.merge_with(engineered_features)
else:
    engineered_data = filtered_data


logging.info("Quantizing targets")
quantized_data = config.quantize(engineered_data, target_omic="DRUGS", IC50s=IC50s)

x = quantized_data.to_pandas(omic='DRUGS')
print(x)



final_data = quantized_data.normalize().optimize_formats()

# logging.info("Visualizing distributions")
# config.visualize_dataset(final_data, mode='post')


logging.info("Getting optimized models")
optimal_algos = config.get_models(dataset=final_data)
config.save(to_save=optimal_algos, name='optimal_algos_complete')

algos_dict, results_prim = config.get_best_algos(optimal_algos)

config.show_results(results_prim)







#%%

logging.info("Creating best stacks")
best_stacks, results_sec = config.get_best_stacks(models=algos_dict, dataset=engineered_data)
algos_dict2, _ = config.get_best_algos(optimal_algos, mode='over')
over_stacks, results_over = config.get_best_stacks(models=algos_dict2, dataset=engineered_data, tag='_over')

results = config.show_results([results_prim, results_sec, results_over])

config.save(to_save=best_stacks, name='stack_results')
config.save(to_save=over_stacks, name='stack_results2')
config.save(to_save=results, name='overall_results')

print('DONE')


# logging.info("Generating results")
# config.evaluate_stacks(best_stacks)


# logging.info("First-level models optimized")
# logging.info("Splitting dataset for cross-validation")
# folds_list = config.split(dataset=engineered_data, split_type='outer')

# for train_index_outer, test_index_outer in folds_list:
# 	training_data, test_data = filtered_data.split(train_index_outer, test_index_outer)
# 	
# 	logging.info("Selecting subsets for feature engineering")
# 	selected_training_subset, selected_test_subset = config.select_subsets(training_data, test_data)
# 	
# 	logging.info("Engineering features")
# 	engineered_training_features = config.engineer_features(selected_training_subset)
# 	engineered_test_features = config.engineer_features(selected_test_subset)
# 	
# 	logging.info("Merging engineered features")
# 	engineered_training_data = training_data.merge_with(engineered_training_features)
# 	engineered_test_data = test_data.merge_with(engineered_test_features)
# 	
# 	logging.info("Imputing missing data")
# 	engineered_training_data = engineered_training_data.impute()
# 	engineered_test_data = engineered_test_data.impute()
# 	
# # 	logging.info("Getting optimized models")
# # 	optimal_algos = config.get_optimized_models(dataset=engineered_training_data)
# 	
# 	
# 	logging.info("Splitting dataset for ensembling optimization")
# 	kfold_inner = config.split(dataset=filtered_data, split_type='inner')
# 	
# 	logging.info("Creating best stacks")
# 	best_stack = config.get_best_stack(dataset=engineered_training_data, algorithms=optimal_algos, kfold=kfold_inner)
# 	
# 	logging.info("Generating results")
# 	config.generate_results(best_stack, optimal_algos, engineered_test_data)
