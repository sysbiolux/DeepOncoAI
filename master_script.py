# -*- coding: utf-8 -*-
'''
master script for analysis
'''
### IMPORTS

# from DBM_toolbox.modeling.get_optimized_models import get_optimal_models
import logging

from config import Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')

config = Config()

logging.info("Reading data")
data = config.read_data()

logging.info("Creating filters")
filters = config.create_filters(data)

logging.info("Applying filters")
filtered_data = data.apply_filters(filters=filters)

logging.info("Selecting subsets for feature engineering")
selected_subset = config.select_subsets(filtered_data)

logging.info("Engineering features")
engineered_features = config.engineer_features(selected_subset)

logging.info("Merging engineered features")
engineered_data = filtered_data.merge_with(engineered_features)

logging.info("Quantizing targets")
filtered_data = filtered_data.quantize(target_omic="DRUGS")

logging.info("Splitting dataset for cross-validation")
outer_index = config.split(dataset=filtered_data, split_type='outer')

logging.info("Getting optimized models")
optimal_algos = config.get_optimized_models(dataset=engineered_data)

for train_index_outer, test_index_outer in outer_index:
	training_data, test_data = filtered_data.split(train_index_outer, test_index_outer)
	
	logging.info("Selecting subsets for feature engineering")
	selected_training_subset, selected_test_subset = config.select_subsets(training_data, test_data)
	
	logging.info("Engineering features")
	engineered_training_features = config.engineer_features(selected_training_subset)
	engineered_test_features = config.engineer_features(selected_test_subset)
	
	logging.info("Merging engineered features")
	engineered_training_data = training_data.merge_with(engineered_training_features)
	engineered_test_data = test_data.merge_with(engineered_test_features)
	
	logging.info("Imputing missing data")
	engineered_training_data = engineered_training_data.impute()
	engineered_test_data = engineered_test_data.impute()
	
# 	logging.info("Getting optimized models")
# 	optimal_algos = config.get_optimized_models(dataset=engineered_training_data)
	
	
	logging.info("Splitting dataset for ensembling optimization")
	kfold_inner = config.split(dataset=filtered_data, split_type='inner')
	
	logging.info("Creating best stack")
	best_stack = config.get_best_stack(dataset=engineered_training_data, algorithms=optimal_algos, kfold=kfold_inner)
	
	logging.info("Generating results")
	config.generate_results(best_stack, optimal_algos, engineered_test_data)
