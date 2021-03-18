import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from DBM_toolbox.data_manipulation import load_data, rule, preprocessing
from DBM_toolbox.data_manipulation import dataset_class, filter_class
from DBM_toolbox.feature_engineering.predictors import combinations, components
from DBM_toolbox.modeling import optimized_models, stacking


parse_filter_dict = {'sample_completeness': lambda this_filter, omic, database: filter_class.KeepDenseRowsFilter(completeness_threshold=this_filter['threshold']),
					 'feature_completeness': lambda this_filter, omic, database: rule.ColumnDensityRule(completeness_threshold=this_filter['threshold'], omic=omic, database=database),
					 'feature_variance': lambda this_filter, omic, database: rule.HighestVarianceRule(fraction=this_filter['fraction_retained'], omic=omic, database=database),
					 'cross-correlation': lambda this_filter, omic, database: rule.CrossCorrelationRule(correlation_threshold=this_filter['correlation_threshold'], omic=omic, database=database)
}

parse_selection_dict = {'importance': lambda selection, omic, database: rule.FeatureImportanceRule(fraction=selection['fraction_selected'], omic=omic, database=database),
						'predictivity': lambda selection, omic, database: rule.FeaturePredictivityRule(fraction=selection['fraction_selected'], omic=omic, database=database)
}

parse_transformation_dict = {'PCA': lambda dataframe, transformation, omic, database: components.get_PCs(dataframe, n_components=transformation['n_components'], label=omic+'_'+database),
							 'ICA': lambda dataframe, transformation, omic, database: components.get_ICs(dataframe, n_components=transformation['n_components'], label=omic+'_'+database),
							 'RP': lambda dataframe, transformation, omic, database: components.get_RPCs(dataframe, n_components=transformation['n_components'], label=omic+'_'+database),
							 'TSNE': lambda dataframe, transformation, omic, database: components.get_TSNEs(dataframe, n_components=transformation['n_components'], label=omic+'_'+database),
							 'Poly': lambda dataframe, transformation, omic, database: combinations.get_polynomials(dataframe, degree=transformation['degree'], label=omic+'_'+database),
							 'OR': lambda dataframe, transformation, omic, database: combinations.get_boolean_or(dataframe)
}

def parse_filter(this_filter: dict, omic: str, database: str):
	try:
		if 'enabled' in this_filter and not this_filter['enabled']:
			return None
		return parse_filter_dict[this_filter['name']](this_filter, omic, database)
	except KeyError:
		raise ValueError(f"Did not recognize filter with {this_filter['name']}")

def parse_selection(selection: dict, omic: str, database: str):
	try:
		if 'enabled' in selection and not selection['enabled']:
			return None
		return parse_selection_dict[selection['name']](selection, omic, database)
	except KeyError:
		raise ValueError(f"Did not recognize engineering selection with {selection['name']}")

def parse_transformations(dataframe, transformation: dict, omic: str, database: str):
	try:
		if 'enabled' in transformation and not transformation['enabled']:
			return None
		return parse_transformation_dict[transformation['name']](dataframe, transformation, omic, database)
	except KeyError:
		raise ValueError(f"Did not recognize transformation with {transformation['name']}")


class Config:
	def __init__(self):
		with open('config.yaml') as f:
			self.raw_dict = yaml.load(f, Loader=yaml.FullLoader)

	def read_data(self):
		nrows = self.raw_dict['data'].get("maximum_rows", None)
		omic = self.raw_dict['data']['omics'][0]
		logging.info(f"Adding dataset {omic['name']} from database {omic['database']}")
		full_dataset = load_data.read_data('data', omic=omic['name'], database=omic['database'])
		for omic in self.raw_dict['data']['omics'][1:]:
			logging.info(f"Adding dataset {omic['name']} from database {omic['database']}")
			additional_dataset = load_data.read_data('data', omic=omic['name'], database=omic['database'], nrows=nrows)
			full_dataset = full_dataset.merge_with(additional_dataset)
		
		targets = self.raw_dict['data']['targets']
		for target in targets:
			target_metric = target['responses']
			target_name = target['target_drug_name']
			logging.info(f"Adding target {target_name} from database {omic['database']}")
			additional_dataset = load_data.read_data('data', omic=target['name'], database=target['database'], keywords=[target_name, target_metric])
			for item in target['normalization']:
				if 'enabled' in item and not item['enabled']:
					pass
				else:
					if item['name'] == 'unit':
						additional_dataset = additional_dataset.normalize()
			full_dataset = full_dataset.merge_with(additional_dataset)
		return full_dataset

	def create_filters(self, dataset):
		omics = self.raw_dict['data']['omics']
		targets = self.raw_dict['data']['targets']
		filters = []
		for omic in omics:
			for this_filter in omic['filtering']:
				new_rule = parse_filter(this_filter, omic['name'], omic['database'])
				if new_rule is not None:
					logging.info(f"Creating filter {this_filter['name']} for {omic['name']}_{omic['database']}")
					if this_filter['name'] == 'sample_completeness': #TODO: this does not look pretty, the fact that sample completeness is a non-transferable filter makes it not have the same "create_filter" as the others so excetptions have to be created.
						new_filter = new_rule
					else:
						new_filter = new_rule.create_filter(dataset)
						if new_filter is not None:
							filters.append(new_filter)
		for target in targets:
			for this_filter in target['filtering']:
				new_rule = parse_filter(this_filter, target['name'], target['database'])
				if new_rule is not None:
					logging.info(f"Creating filter {this_filter['name']} for {target['name']}_{target['database']}")
					if this_filter['name'] == 'sample_completeness': #TODO: this does not look pretty, the fact that sample completeness is a non-transferable filter makes it not have the same "create_filter" as the others so excetptions have to be created.
						new_filter = new_rule
					else:
						new_filter = new_rule.create_filter(dataset)
						if new_filter is not None:
							filters.append(new_filter)
		return filters

	def select_subsets(self, datasets):
		if isinstance(datasets, list): #TODO: this is not clean anymore. find another way to accept both single datasets and transfer from one to another
			training_dataset = datasets[0]
			test_dataset = datasets[1]
		else:
			training_dataset = datasets
		omics = self.raw_dict['data']['omics']
		training_dataframe = training_dataset.to_pandas()
		target_name = self.raw_dict['data']['targets'][0]['target_drug_name'] #TODO!!!: this should loop over all targets. the [0] is a temporary fix when there is only one
		response = self.raw_dict['data']['targets'][0]['responses']
		try:
			target = training_dataframe[target_name + '_' + response]
		except:
			raise ValueError(f"Did not recognize target name with {target_name}_{response}")
		selected_training_subset = None
		selected_test_subset = None
		for omic in omics:
			database = omic['database']
			for selection in omic['feature_engineering']['feature_selection']:
				print('**************')
				logging.info(f"Creating selection for {omic['name']}_{database}")
				new_selection = parse_selection(selection=selection, omic=omic['name'], database=database)
				print(new_selection)
				if new_selection is not None:
					this_dataset = dataset_class.Dataset(dataframe=training_dataset.to_pandas(omic=omic['name'], database=database), omic=omic['name'], database=database)
					print(this_dataset)
					new_filter = new_selection.create_filter(dataset=this_dataset, target_df=target)
					print(new_filter)
					logging.info(f"Applying selection for {omic['name']}_{database}")
					new_training_subset = this_dataset.apply_filters([new_filter])
					print(new_training_subset)
					if isinstance(datasets, list):
						this_test_dataset = dataset_class.Dataset(dataframe=test_dataset.to_pandas(omic=omic['name'], database=database), omic=omic['name'], database=database)
						new_test_subset = this_test_dataset.apply_filters([new_filter])
					if selected_training_subset is not None:
						columns_to_add = new_training_subset.dataframe.columns.difference(selected_training_subset.dataframe.columns)
						to_merge = dataset_class.Dataset(dataframe=new_training_subset.dataframe[columns_to_add],
														omic=new_training_subset.omic[columns_to_add],
														database=new_training_subset.database[columns_to_add])
						selected_training_subset = selected_training_subset.merge_with(to_merge)
						if isinstance(datasets, list):
							columns_to_add = new_test_subset.dataframe.columns.difference(selected_test_subset.dataframe.columns)
							selected_test_subset = selected_test_subset.merge_with(columns_to_add)
					else:
						selected_training_subset = new_training_subset
						if isinstance(datasets, list):
							selected_test_subset = new_test_subset
				else:
					logging.info("inactive")
		if isinstance(datasets, list):
			return selected_training_subset, selected_test_subset
		else:
			return selected_training_subset

	def engineer_features(self, dataset):
		omics = self.raw_dict['data']['omics']
		
		engineered_features = None
		for omic in omics:
			print(omic)
			dataframe = dataset.to_pandas(omic=omic['name'], database=omic['database'])
			print(dataframe)
			transformations_dict = omic['feature_engineering']['transformations']
			for transformation in transformations_dict:
				logging.info(f"Engineering {transformation['name']} features for {omic['name']}_{omic['database']}")
				new_features = parse_transformations(dataframe=dataframe, transformation=transformation, omic=omic['name'], database=omic['database'])
				if new_features is not None:
					if engineered_features is not None:
						engineered_features = engineered_features.merge_with(new_features)
					else:
						engineered_features = new_features
				else:
					logging.info("inactive")
		using_types = self.raw_dict['modeling']['general']['use_tumor_type']['enabled']
		if using_types:
			new_features = components.get_tumor_type(dataframe)
			if engineered_features is not None:
				engineered_features = engineered_features.merge_with(new_features)
			else:
				engineered_features = new_features
		return engineered_features

	def split(self, dataset, split_type):
		dataframe = dataset.to_pandas()
		modeling_options = self.raw_dict['modeling']['general']
		if split_type == 'outer':
			n_splits = modeling_options['outer_folds']['value']
			random_state = modeling_options['outer_folds']['random_seed']
		elif split_type == 'inner':
			n_splits = modeling_options['inner_folds']['value']
			random_state = modeling_options['inner_folds']['random_seed']
		target_name = self.raw_dict['data']['targets'][0]['target_drug_name']
		response = self.raw_dict['data']['targets'][0]['responses']
		try:
			target = dataframe[target_name + '_' + response]
		except:
			raise ValueError(f"Did not recognize target name with {target_name}_{response}")
		splitting = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
		split_index = splitting.split(dataframe, target)
		return split_index
	
	def get_optimized_models(self, dataset, algos=None):
		omic_list = list(dict.fromkeys(dataset.omic.tolist()))
		target_list = []
		metric = self.raw_dict['modeling']['general']['metric']
		for item in self.raw_dict['data']['targets']:
			target_list.append(item['name'])
		modeling_options = self.raw_dict['modeling']['general']['search_depth']
		if modeling_options['enabled']:
			depth = modeling_options['value']
		else:
			depth = None
		
		
		targets_colnames = dataset.dataframe.loc[:, dataset.omic.str.startswith('DRUGS')].columns
		results = dict()
		
		for this_target in targets_colnames:
			# TODO: there should be a better way to do this, this depends on the exact order of the targets, should be ok but maybe names are better
			results[this_target] = dict()
			this_dataset = dataset.to_binary(target=this_target)
			for this_omic in omic_list:
				if this_omic not in target_list:
					print(this_omic)
					this_data = this_dataset.to_pandas(omic=this_omic)
					logging.info(f"Optimizing models for {this_target} with {this_omic}")
					this_result = optimized_models.bayes_optimize_models(data=this_data, 
																		targets=this_dataset.dataframe[this_target], 
																		n_trials=depth, 
																		algos=algos, 
																		metric=metric)
					
					if this_omic not in results[this_target]:
						results[this_target][this_omic] = this_result
				
		return results
	
	def visualize_optimized_models(self, results):
		
		results_list = dict()
		targets = results.keys()
		
		for target in targets:
			
			omics = list(results[target].keys())
			algos = results[target][omics[0]].keys()
			df = pd.DataFrame(index=algos, columns=omics)
			for omic in omics:
				print(omic)
				for algo in algos:
					print(algo)
					print(results[target])
					print(results[target][omic])
					print(results[target][omic][algo])
					print(results[target][omic][algo]['result'])
					print(df)
					this_result = results[target][omic][algo]['result']
					if this_result is not np.nan:
						df.loc[algo,omic] = this_result['target']
					else:
						df.loc[algo,omic] = np.nan
						
			results_list[target] = df
		
		return results_list
		
	
	def get_best_stack(self, dataset, algorithms, kfold):
		
		
		
		
		
		
		pass

	def generate_results(self, best_stack, optimal_algos, test_dataset):
		pass
