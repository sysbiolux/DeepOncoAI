import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')


# from DBM_toolbox.modeling.get_optimized_models import get_optimal_models

from config import Config


config = Config()

logging.info("Reading data")
data = config.read_data()

logging.info("Creating filters")
filters = config.create_filters(data)

logging.info("Applying filters")
filtered_data = data.apply_filters(filters=filters)

logging.info("Selecting subsets")
selected_subsets = config.select_subsets(filtered_data)

logging.info("Engineering features")
engineered_features = config.engineer_features(selected_subsets)

logging.info("Merging engineered features")
engineered_data = filtered_data.merge_with(engineered_features)

logging.info("Quantizing targets")
engineered_data = engineered_data.quantize(target_omic="DRUGS")

###debug
x = engineered_data.dataframe['AEW541_ActArea']

logging.info("Get optimized models")
optimal_algos = config.get_optimized_models(engineered_data)

print(optimal_algos)

logging.info("Create best stack")
best_stack = config.get_best_stack(engineered_data, optimal_algos)

logging.info("Generate results")
config.generate_results(best_stack, optimal_algos)


# ### DATA VISUALISATION




# ### DATA PREPROCESSING

# # remove features with too much missing data
# column_density_rule_rppa = ColumnDensityRule(density_fraction=0.95, omic='RPPA')
# # column_density_rule_rna = ColumnDensityRule(density_fraction=0.95, omic='RNA')
# # column_density_rule_mirna = ColumnDensityRule(density_fraction=0.95, omic='MIRNA')
# column_density_rule_drugs = ColumnDensityRule(density_fraction=1, omic='DRUGS')

# # remove features with low variance
# highest_variance_rule_rppa = HighestVarianceRule(fraction=0.8, omic='RPPA')
# # highest_variance_rule_rna = HighestVarianceRule(fraction=0.2, omic='RNA')
# # highest_variance_rule_mirna = HighestVarianceRule(fraction=0.2, omic='MIRNA')

# # remove incomplete samples
# density_row_filter = [KeepDenseRowsFilter(density_fraction=0.95)]

# # making and applying filters
# column_density_filters = [column_density_rule_rppa.create_filter(data), 
# # 						  column_density_rule_rna.create_filter(data), 
# 						#   column_density_rule_mirna.create_filter(data), 
# 						  column_density_rule_drugs.create_filter(data)]
# highest_variance_filters = [highest_variance_rule_rppa.create_filter(data)]
# # 							highest_variance_rule_rna.create_filter(data),
# 							# highest_variance_rule_mirna.create_filter(data)]

# filters = density_row_filter + column_density_filters + highest_variance_filters
# filtered_data = data.apply_filters(filters=filters)


# ### imputing and normalizing data
# imputed_data = filtered_data.impute()
# normalized_data = imputed_data.normalize()



# ### FEATURE ENGINEERING


# ### selection of the target drug response
# selected_target = normalized_data.to_pandas(omic='DRUGS').iloc[:,1] # selected target drug responses


# ### subsets based on feature importance
# selection_rule_rppa_1 = FeatureImportanceRule(fraction=0.1, omic='RPPA')
# # selection_rule_rna_1 = FeatureImportanceRule(fraction=0.01, omic='RNA')
# # selection_rule_mirna_1 = FeatureImportanceRule(fraction=0.1, omic='MIRNA')

# selection_filter_rppa_1 = selection_rule_rppa_1.create_filter(normalized_data, selected_target)
# # selection_filter_rna_1 = selection_rule_rna_1.create_filter(normalized_data, selected_target)
# # selection_filter_mirna_1 = selection_rule_mirna_1.create_filter(normalized_data, selected_target)

# most_important_rppa = normalized_data.apply_filters(filters=[selection_filter_rppa_1])
# # most_important_rna = normalized_data.apply_filters(filters=[selection_filter_rna_1])
# # most_important_mirna = normalized_data.apply_filters(filters=[selection_filter_mirna_1])

# ### subsets based on feature predictivity
# selection_rule_rppa_2 = FeaturePredictivityRule(fraction=0.1, omic='RPPA')
# # selection_rule_rna_2 = FeaturePredictivityRule(fraction=0.01, omic='RNA')
# # selection_rule_mirna_2 = FeaturePredictivityRule(fraction=0.1, omic='MIRNA')

# # selection_filter_rppa_2 = selection_rule_rppa_2.create_filter(normalized_data, selected_target)
# # selection_filter_rna_2 = selection_rule_rna_2.create_filter(normalized_data, selected_target)
# # selection_filter_mirna_2 = selection_rule_mirna_2.create_filter(normalized_data, selected_target)

# # most_predictive_rppa = normalized_data.apply_filters(filters=[selection_filter_rppa_2])
# # most_important_rna = normalized_data.apply_filters(filters=[selection_filter_rna_2])
# # most_predictive_mirna = normalized_data.apply_filters(filters=[selection_filter_mirna_2])









# most_important_rppa_PC = most_important_rppa.get_features('PC', target_omic='RPPA', options=[10, 'rppa_imp_PC'])
# # most_predictive_rppa_PC = most_predictive_rppa.get_features('PC', target_omic='RPPA', options=[10, 'rppa_pred_PC'])
# most_important_rppa_IC = most_important_rppa.get_features('IC', target_omic='RPPA', options=[10, 'rppa_imp_IC'])
# # most_predictive_rppa_IC = most_predictive_rppa.get_features('IC', target_omic='RPPA', options=[10, 'rppa_pred_IC'])
# most_important_rppa_TSNE = most_important_rppa.get_features('TSNE', target_omic='RPPA', options=[2, 'rppa_imp_TSNE'])
# # most_predictive_rppa_TSNE = most_predictive_rppa.get_features('TSNE', target_omic='RPPA', options=[2, 'rppa_pred_TSNE'])
# most_important_rppa_POLY = most_important_rppa.get_features('POLY', target_omic='RPPA')
# # most_predictive_rppa_POLY = most_predictive_rppa.get_features('POLY', target_omic='RPPA')
# most_important_rppa_OR = most_important_rppa.get_features('OR', target_omic='RPPA')
# # most_predictive_rppa_OR = most_predictive_rppa.get_features('OR', target_omic='RPPA')

# # most_important_rna_PC = most_important_rna.get_features('PC', target_omic='RNA', options=[10, 'rna_imp_PC'])
# # most_predictive_rna_PC = most_predictive_rna.get_features('PC', target_omic='RNA', options=[10, 'rna_pred_PC'])
# # most_important_rna_IC = most_important_rna.get_features('IC', target_omic='RNA', options=[10, 'rna_imp_IC'])
# # most_predictive_rna_IC = most_predictive_rna.get_features('IC', target_omic='RNA', options=[10, 'rna_pred_IC'])
# # most_important_rna_TSNE = most_important_rna.get_features('TSNE', target_omic='RNA', options=[2, 'rna_imp_TSNE'])
# # most_predictive_rna_TSNE = most_predictive_rna.get_features('TSNE', target_omic='RNA', options=[2, 'rna_pred_TSNE'])
# # most_important_rna_POLY = most_important_rna.get_features('POLY', target_omic='RNA')
# # most_predictive_rna_POLY = most_predictive_rna.get_features('POLY', target_omic='RNA')
# # most_important_rna_OR = most_important_rna.get_features('OR', target_omic='RNA')
# # most_predictive_rna_OR = most_predictive_rna.get_features('OR', target_omic='RNA')

# # most_important_mirna_PC = most_important_mirna.get_features('PC', target_omic='MIRNA', options=[10, 'mirna_imp_PC'])
# # most_predictive_mirna_PC = most_predictive_mirna.get_features('PC', target_omic='MIRNA', options=[10, 'mirna_pred_PC'])
# # most_important_mirna_IC = most_important_mirna.get_features('IC', target_omic='MIRNA', options=[10, 'mirna_imp_IC'])
# # most_predictive_mirna_IC = most_predictive_mirna.get_features('IC', target_omic='MIRNA', options=[10, 'mirna_pred_IC'])
# # most_important_mirna_TSNE = most_important_mirna.get_features('TSNE', target_omic='MIRNA', options=[2, 'mirna_imp_TSNE'])
# # most_predictive_mirna_TSNE = most_predictive_mirna.get_features('TSNE', target_omic='MIRNA', options=[2, 'mirna_pred_TSNE'])
# # most_important_mirna_POLY = most_important_mirna.get_features('POLY', target_omic='MIRNA')
# # most_predictive_mirna_POLY = most_predictive_mirna.get_features('POLY', target_omic='MIRNA')
# # most_important_mirna_OR = most_important_mirna.get_features('OR', target_omic='MIRNA')
# # most_predictive_mirna_OR = most_predictive_mirna.get_features('OR', target_omic='MIRNA')

# engineered_features_rppa = most_important_rppa_PC.merge_with([
# 	# most_predictive_rppa_PC,
# 	most_important_rppa_IC,
# 	# most_predictive_rppa_IC,
# 	most_important_rppa_TSNE,
# 	# most_predictive_rppa_TSNE,
# 	most_important_rppa_POLY,
# 	# most_predictive_rppa_POLY,
# 	most_important_rppa_OR])
# 	# most_predictive_rppa_OR])
# 	
# # engineered_features_rna = most_important_rna_PC.merge_with(
# 	# most_predictive_rna_PC
# 	# most_important_rna_IC
# 	# most_predictive_rna_IC
# 	# most_important_rna_TSNE
# 	# most_predictive_rna_TSNE
# 	# most_important_rna_POLY
# 	# most_predictive_rna_POLY
# 	# most_important_rna_OR
# 	# most_predictive_rna_OR)
# 	
# # engineered_features_mirna = most_important_mirna_PC.merge_with([
# # 	most_predictive_mirna_PC,
# # 	most_important_mirna_IC,
# # 	most_predictive_mirna_IC,
# # 	most_important_mirna_TSNE,
# # 	most_predictive_mirna_TSNE,
# # 	most_important_mirna_POLY,
# # 	most_predictive_mirna_POLY,
# # 	most_important_mirna_OR,
# # 	most_predictive_mirna_OR])

# tumor_types = normalized_data.get_features('TYPE')

# data = normalized_data.merge_with([engineered_features_rppa, tumor_types])
# # data = normalized_data.merge_with([engineered_features_rppa, engineered_features_mirna, tumor_types])

# data = data.quantize(target_omic='DRUGS', quantiles=[0.333, 0.667])

# print(data.dataframe.shape)
# ### remove duplicated columns
# datadd = data.dataframe.loc[:,~data.dataframe.columns.duplicated()]

# ### DATA VISUALISATION BEFORE MODELING

# print(selected_target)
# models = get_optimal_models(datadd, selected_target, n_trials=5)
# print(models)

### MODELING




### RESULT ANALYSIS



### 