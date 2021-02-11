import yaml
import logging
from sklearn.model_selection import StratifiedKFold

from DBM_toolbox.data_manipulation import load_data, rule, preprocessing
from DBM_toolbox.feature_engineering.targets import ternarization
from DBM_toolbox.data_manipulation import dataset_class, filter_class
from DBM_toolbox.feature_engineering.predictors import combinations, components
from DBM_toolbox.modeling import optimized_models, stacking


parse_filter_dict = {'sample_completeness': lambda this_filter, omic, database: filter_class.KeepDenseRowsFilter(completeness_threshold=this_filter['threshold']),
					 'feature_completeness': lambda this_filter, omic, database: rule.ColumnDensityRule(completeness_threshold=this_filter['threshold'], omic=omic, database=database),
					 'feature_variance': lambda this_filter, omic, database: rule.HighestVarianceRule(fraction=this_filter['fraction_retained'], omic=omic, database=database)
}

parse_selection_dict = {'importance': lambda selection, omic, database: rule.FeatureImportanceRule(fraction=selection['fraction_selected'], omic=omic, database=database),
						  'predictivity': lambda selection, omic, database: rule.FeaturePredictivityRule(fraction=selection['fraction_selected'], omic=omic, database=database)
}

parse_transformation_dict = {'PCA': lambda dataframe, transformation, omic: components.get_PCs(dataframe, n_components=transformation['n_components']),
							 'ICA': lambda dataframe, transformation, omic: components.get_ICs(dataframe, n_components=transformation['n_components']),
							 'RP': lambda dataframe, transformation, omic: components.get_RPCs(dataframe, n_components=transformation['n_components']),
							 'TSNE': lambda dataframe, transformation, omic: components.get_TSNEs(dataframe, n_components=transformation['n_components']),
							 'Poly': lambda dataframe, transformation, omic: combinations.get_polynomials(dataframe, degree=transformation['degree']),
							 'OR': lambda dataframe, transformation, omic: combinations.get_boolean_or(dataframe)
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
		return parse_transformation_dict[transformation['name']](dataframe, transformation, omic)
	except KeyError:
		raise ValueError(f"Did not recognize transformation with {transformation['name']}")


class Config:
	def __init__(self):
		with open('config.yaml') as f:
			self.raw_dict = yaml.load(f, Loader=yaml.FullLoader)

	def read_data(self):
		omic = self.raw_dict['data']['omics'][0]
		full_dataset = load_data.read_data('data', omic=omic['name'], database=omic['database'])
		for omic in self.raw_dict['data']['omics'][1:]:
			additional_dataset = load_data.read_data('data', omic=omic['name'], database=omic['database'])
			full_dataset = full_dataset.merge_with(additional_dataset)
		
		targets = self.raw_dict['data']['targets']
		for target in targets:
			target_metric = target['responses']
			target_name = target['target_drug_name']
			additional_dataset = load_data.read_data('data', omic=target['name'], database=target['database'], keywords=[target_name, target_metric])
			# TODO: include here quantization
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
		filters = []
		for omic in omics:
			for this_filter in omic['filtering']:
				new_rule = parse_filter(this_filter, omic['name'], omic['database'])
				if new_rule is not None:
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
# 		test_dataframe = test_dataset.to_pandas()
		target_name = self.raw_dict['data']['targets'][0]['target_drug_name'] #TODO: this should loop over all targets. the [0] is a temporary fix when there is only one
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
				new_selection = parse_selection(selection=selection, omic=omic['name'], database=database)
				if new_selection is not None:
					logging.info(f"Creating filter for {omic['name']}_{database}")
					new_filter = new_selection.create_filter(dataset=training_dataset, target_df=target)
					new_training_subset = training_dataset.apply_filters([new_filter])
					if isinstance(datasets, list):
						new_test_subset = test_dataset.apply_filters([new_filter])
					if selected_training_subset is not None:
						selected_training_subset = selected_training_subset.merge_with(new_training_subset)
						if isinstance(datasets, list):
							selected_test_subset = selected_test_subset.merge_with(new_test_subset)
					else:
						selected_training_subset = new_training_subset
						if isinstance(datasets, list):
							selected_test_subset = new_test_subset
		if isinstance(datasets, list):
			return selected_training_subset, selected_test_subset
		else:
			return selected_training_subset

	def engineer_features(self, dataset):
		omics = self.raw_dict['data']['omics']
		dataframe = dataset.to_pandas()
		engineered_features = None
		database = dataset.database
		for omic in omics:
			transformations_dict = omic['feature_engineering']['transformations']
			for transformation in transformations_dict:
				new_features = parse_transformations(dataframe=dataframe, transformation=transformation, omic=omic, database=database)
				if new_features is not None:
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
		
		omic_list = self.raw_dict['data']['omics']
		modeling_options = self.raw_dict['modeling']['general']['search_depth']
		if modeling_options['enabled']:
			depth = modeling_options['value']
		else:
			depth = None
		
		targets = dataset.to_pandas(omic='DRUGS')
		results = dict()
		
		for this_target in targets.columns:
			# TODO: there should be a better way to do this, this depends on the exact order of the targets, should be ok but maybe names are better
			for this_omic in omic_list:
				this_data = dataset.to_pandas(omic=this_omic['name'], database=this_omic['database'])
				logging.info(f"Optimized models for {this_target} with {this_omic['name']} from {this_omic['database']}")
				this_result = optimized_models.bayes_optimize_models(data=this_data, targets=dataset.dataframe[this_target], n_trials=depth, algos=algos)
				
				omic_db = this_omic['name'] + '_' + this_omic['database']
				results[omic_db][this_target] = this_result
				
				
	def get_best_stack(self, dataset, algorithms, kfold):
		
		
		
		
		
		
		pass

	def generate_results(self, best_stack, optimal_algos, test_dataset):
		pass









































