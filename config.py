import yaml
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
import xgboost as xgb
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt

from DBM_toolbox.data_manipulation import load_data, rule,  preprocessing
from DBM_toolbox.data_manipulation import dataset_class, filter_class
from DBM_toolbox.feature_engineering.predictors import combinations, components
from DBM_toolbox.modeling import optimized_models, stacking
from DBM_toolbox.plotting import plot_eda

parse_filter_dict = {'sample_completeness': lambda this_filter, omic, database: filter_class.KeepDenseRowsFilter(completeness_threshold=this_filter['threshold']),
					 'feature_completeness': lambda this_filter, omic, database: rule.ColumnDensityRule(completeness_threshold=this_filter['threshold'], omic=omic, database=database),
					 'feature_variance': lambda this_filter, omic, database: rule.HighestVarianceRule(fraction=this_filter['fraction_retained'], omic=omic, database=database),
					 'cross-correlation': lambda this_filter, omic, database: rule.CrossCorrelationRule(correlation_threshold=this_filter['correlation_threshold'], omic=omic, database=database)
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
	'''
	Generates a filter Rule if the option is enabled in the config file
	'''
	try:
		if 'enabled' in this_filter and not this_filter['enabled']:
			return None
		return parse_filter_dict[this_filter['name']](this_filter, omic, database)
	except KeyError:
		raise ValueError(f"Did not recognize filter with {this_filter['name']}")

def parse_selection(selection: dict, omic: str, database: str):
	'''
	Generates a selection Rule if the option is enabled in the config file
	'''
	try:
		if 'enabled' in selection and not selection['enabled']:
			return None
		return parse_selection_dict[selection['name']](selection, omic, database)
	except KeyError:
		raise ValueError(f"Did not recognize engineering selection with {selection['name']}")

def parse_transformations(dataframe, transformation: dict, omic: str, database: str):
	'''
	Generates and computes a transformation if the option is enabled in the config file
	'''
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
		'''
		Reads the data (omics and targets) according to the config file and assembles a Dataset
		'''
		print('Reading data...')
		nrows = self.raw_dict['data'].get("maximum_rows", None)
		omic = self.raw_dict['data']['omics'][0]
		print('Loading ' + omic['name'] + ' from ' + omic['database'])
		full_dataset = load_data.read_data('data', omic=omic['name'], database=omic['database'])
		for omic in self.raw_dict['data']['omics'][1:]:
			print('Loading ' + omic['name'] + ' from ' + omic['database'])
			additional_dataset = load_data.read_data('data', 
													omic=omic['name'],
													database=omic['database'],
													nrows=nrows)
			full_dataset = full_dataset.merge_with(additional_dataset)
		
		targets = self.raw_dict['data']['targets']
		for target in targets:
			target_metric = target['responses']
			target_name = target['target_drug_name']
			print('Loading ' + target['name'] + ' from ' + target['database'])
			additional_dataset = load_data.read_data('data',
													omic=target['name'],
													database=target['database'],
													keywords=[target_name,
													target_metric])
			full_dataset = full_dataset.merge_with(additional_dataset)
		return full_dataset
	
	def split(self, dataset: dataset_class.Dataset, target_name: str, split_type: str ='outer'):
		'''
		Splits the Dataset according to the split type indicated in the config file
		and returns the list of training and testing indices
		'''
		if split_type == 'outer':
			split_txt = 'outer_folds'
		elif split_type == 'inner':
			split_txt = 'inner'
		else:
			raise ValueError('split type should be either "outer" or "inner"')
		n_splits = self.raw_dict['modeling']['general'][split_txt]['value']
		dataframe = dataset.to_pandas()
		target = dataframe[target_name]
		dataframe.drop(target_name)
		
		xval = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
		folds_list = xval.split(dataframe, target)
		return folds_list

	def create_filters(self, dataset: dataset_class.Dataset):
		'''
		Computes a filter for each omic based on the config file specifications
		Returns the list of filters
		'''
		print('Creating filters...')
		omics = self.raw_dict['data']['omics']
		filters = []
		for omic in omics:
			for this_filter in omic['filtering']:
				print('Creating filter ' + this_filter['name'] + ' for ' + omic['database'] + '/' + omic['name'])
				new_rule = parse_filter(this_filter, omic['name'], omic['database'])
				if new_rule is not None:
					if this_filter['name'] == 'sample_completeness': #TODO: this does not look pretty, the fact that sample completeness is a non-transferable filter makes it not have the same "create_filter" as the others so excetptions have to be created.
						new_filter = new_rule
					else:
						new_filter = new_rule.create_filter(dataset)
						if new_filter is not None:
							filters.append(new_filter)
		return filters

	def select_subsets(self, datasets: list):
		'''
		Selects a subset of the features based on either predictivity or importance or both
		Returns a Dataset
		'''
		print('Selecting subsets...')
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
				logging.info(f"Creating selection for {omic['name']}_{database}")
				print('Selecting within ' + omic['name'] + '/' + database)
				new_selection = parse_selection(selection=selection, omic=omic['name'], database=database)
				if new_selection is not None:
					this_dataset = dataset_class.Dataset(dataframe=training_dataset.to_pandas(omic=omic['name'], database=database), 
														omic=omic['name'],
														database=database)
					new_filter = new_selection.create_filter(dataset=this_dataset, target_dataframe=target)
					logging.info(f"Applying selection for {omic['name']}_{database}")
					new_training_subset = this_dataset.apply_filters([new_filter])
					if isinstance(datasets, list):
						this_test_dataset = dataset_class.Dataset(dataframe=test_dataset.to_pandas(omic=omic['name'], database=database),
																omic=omic['name'],
																database=database)
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

	def engineer_features(self, dataset: dataset_class.Dataset):
		'''
		Applies transformations (PCA, TSNE, combinations) to a dataset and 
		returns the dataset
		'''
		print('Engineering features...')
		omics = self.raw_dict['data']['omics']
		dataframe = dataset.to_pandas()
		engineered_features = None
		database = dataset.database
		for omic in omics:
			transformations_dict = omic['feature_engineering']['transformations']
			for transformation in transformations_dict:
				logging.info(f"Engineering {transformation['name']} for {omic['name']} in {omic['database']}")
				print('Engineering ' + transformation['name'] + ' for ' + omic['name'] + '/' + omic['database'])
				new_features = parse_transformations(dataframe=dataframe, transformation=transformation, omic=omic, database=database)
				if new_features is not None:
					new_features = new_features.remove_constants()
					if engineered_features is not None:
						engineered_features = engineered_features.merge_with(new_features)
					else:
						engineered_features = new_features
		
		use_type = self.raw_dict['modeling']['general']['use_tumor_type']
		if use_type['enabled'] == True:
			logging.info("Retrieving tumor types")
			dataframe_tumors = preprocessing.get_tumor_type(dataframe)
			tumor_dataset = dataset_class.Dataset(dataframe=dataframe_tumors, omic='TYPE', database='OWN').remove_constants()
			engineered_features = engineered_features.merge_with(tumor_dataset)
		
		return engineered_features


	def get_models(self, dataset: dataset_class.Dataset, method: str=None, algos: list=None):
		'''
		Optimizes a set of models by retrieving omics and targets from the comfig files
		Bayesian hyperparameter optimization is performed for each model, predicting each target with each omic.
		Returns the a dictionary of optimized models and their performances 
		'''
		# TODO: track algos from the config file
		print('Computing models')
		targets_list = []
		method = self.raw_dict['modeling']['general']['first_level_models']
		metric = self.raw_dict['modeling']['general']['metric']
		for item in self.raw_dict['data']['targets']:
			targets_list.append(item['name'].split('_')[0])
		targets_list = list(set(targets_list))
		omics_list = list(dict.fromkeys(dataset.omic.tolist()))
		for target in targets_list:
			omics_list.remove(target)
		modeling_options = self.raw_dict['modeling']['general']['search_depth']
		if modeling_options['enabled']:
			depth = modeling_options['value']
		else:
			depth = None
		
		targets_colnames = []
		for root in targets_list:
			items = dataset.dataframe.loc[:, dataset.omic.str.startswith(root)].columns
			for item in items:
				targets_colnames.append(item)
		results = dict()
		
		for this_target in targets_colnames:
			# TODO: there should be a better way to do this, this depends on the exact order of the targets, should be ok but maybe names are better
			results[this_target] = dict()
			this_dataset = dataset.to_binary(target=this_target)
			
			complete_dataset = this_dataset.extract(omics_list=omics_list, databases_list=[]).to_pandas()
			if method == 'optimize':
				logging.info(f"*** Optimizing models for {this_target} with the complete set of predictors")
				print('Optimizing models for '+ this_target + ' with the complete set of predictors')
				this_result = optimized_models.bayes_optimize_models(data=complete_dataset, 
																	targets=this_dataset.dataframe[this_target], 
																	n_trials=depth, 
																	algos=algos, 
																	metric=metric)
				results[this_target]['complete'] = this_result
				
				for this_omic in omics_list:
					this_data = this_dataset.to_pandas(omic=this_omic)
					logging.info(f"*** Optimizing models for {this_target} with {this_omic}")
					print('Optimizing models for ' + this_target + ' with ' + this_omic)
					this_result = optimized_models.bayes_optimize_models(data=this_data, 
																		targets=this_dataset.dataframe[this_target], 
																		n_trials=depth, 
																		algos=algos, 
																		metric=metric)
					
					if this_omic not in results[this_target]:
						results[this_target][this_omic] = this_result
			elif method == 'standard':
				logging.info(f"*** Computing standard models for {this_target} with the complete set of predictors")
				print('Computing standard models for '+ this_target + ' with the complete set of predictors')
				this_result = optimized_models.get_standard_models(data=complete_dataset, 
																	targets=this_dataset.dataframe[this_target], 
																	algos=algos, 
																	metric=metric)
				results[this_target]['complete'] = this_result
				
				for this_omic in omics_list:
					this_data = this_dataset.to_pandas(omic=this_omic)
					logging.info(f"*** Computing standard models for {this_target} with {this_omic}")
					print('Computing standard models for ' + this_target + ' with ' + this_omic)
					this_result = optimized_models.get_standard_models(data=this_data, 
																		targets=this_dataset.dataframe[this_target], 
																		algos=algos, 
																		metric=metric)
					if this_omic not in results[this_target]:
						results[this_target][this_omic] = this_result
				
				
		return results

	def get_best_algos(self, optimal_algos:dict, mode='standard'):
		'''
		Compares the results of different algorithms for the same target with the same omic type, 
		looks for the highest performance and returns a dictionary of models
		'''
		logging.info('Selecting best algorithms')
		options = self.raw_dict['modeling']['ensembling']
		n_models = options['n_models']
		#for each target
		models = dict()
		targets = optimal_algos.keys()
		results_df = pd.DataFrame(columns=['target', 'omic', 'algo', 'perf', 'estim'])
		for target in targets:
			print(target)
			models[target] = dict()
			#ranked list of algos
			if mode == 'standard':
				omics = optimal_algos[target].keys()
			elif mode == 'over':
				omics = ['complete']
				n_models = len(optimal_algos[target]['complete'].keys())
			for omic in omics:
				algos = optimal_algos[target][omic]
				models[target][omic] = []
				for algo in algos:
					i = optimal_algos[target][omic][algo]
					estimator = i['estimator']
					try:
						result = i['result']
					except:
						result = np.nan
					results_df = results_df.append(pd.Series([target, omic, algo, result, estimator],
													index=results_df.columns),
													ignore_index=True)
				# select the best one
				omic_results = results_df[(results_df['target']==target) & (results_df['omic']==omic)]
				best = omic_results.sort_values(by='perf', ascending=False).iloc[0:min(n_models, omic_results.shape[0]), :]
				models[target][omic].append(best['estim'])
		
		return models, results_df
	
	def get_best_stacks(self, models: dict, dataset: dataset_class.Dataset, tag=None):
		'''
		Computes stacks for each target with two stacking types:
			- lean: only with predictions of the first-level models
			- full: with predictions of the first-level models and the original data
		'''
		options = self.raw_dict['modeling']['ensembling']
		metric = self.raw_dict['modeling']['general']['metric']
		folds = self.raw_dict['modeling']['general']['inner_folds']['value']
		seed = self.raw_dict['modeling']['general']['inner_folds']['random_seed']
		n_models = options['n_models']
		targets_list = list()
		if tag == None:
			tag = ''
		for item in self.raw_dict['data']['targets']:
			this_name = item['target_drug_name'] + '_' + item['responses']
			targets_list.append(this_name)
		if options['metalearner'] == 'XGBoost':
			final_model = xgb.XGBClassifier(n_estimators=100,
											max_depth=6,
											random_state=seed,
											learning_rate=0.1,
											colsample_bytree= 0.9,
											subsample=0.9,
											n_jobs=-1,
											)
		elif options['metalearner'] == 'Logistic':
			final_model = LogisticRegressionCV(random_state=42, max_iter=1000, n_jobs=-1)
		elif options['metalearner'] == 'SVM':
			final_model = SVC(kernel='linear', probability=True, random_state=42)
		else:
			raise ValueError('Metalearner ' + options['metalearner'] + ' not recognized')
		
		folds = min(folds, len(dataset.dataframe.index))
		best_stacks = stacking.compute_stacks(dataset, models, final_model, targets_list, metric, folds, seed)
		results_df = pd.DataFrame(columns=['target', 'omic', 'algo', 'perf', 'estim'])
		for target in best_stacks.keys():
			scores = best_stacks[target]['scores']
			for stack_type in scores.index:
				results_df = results_df.append(pd.Series([target, ('stack' + tag), stack_type, scores[stack_type], final_model],
													index=results_df.columns),
													ignore_index=True)

		return best_stacks, results_df
		
	def get_over_stacks(self, models: dict, dataset: dataset_class.Dataset):
		
		pass
	
	def show_results(self, datasets: list):
		pass
		results = pd.concat(datasets, ignore_index=True)
		return results
	
	def visualize_dataset(self, dataset):
		# TODO: get visualization options from the config file
		omics = dataset.omic
		databases = dataset.database
		for database in databases:
			for omic in omics:
				logging.info(f"plotting info for {omic} in {database}")
				dataframe = dataset.to_pandas(omic=omic, database=database)
				if len(dataframe.columns) <= 20:
					plot_eda.plot_eda(dataframe)
				else:
					keep_plotting = True
					while keep_plotting:
						plot_eda.plot_eda(dataframe.iloc[:, :20])
						dataframe = dataframe.iloc[:, 20:]
						
						
						
	def evaluate_stacks(best_stacks):
		pass

	def save(self, to_save=[], name='file'):
		date = datetime.now()
		timestamp = date.strftime("%Y-%m-%d-%H-%M-%S-%f")
		filename = name + '_' + timestamp + ".pkl"
		f = open(filename,"wb")
		pickle.dump(to_save,f)
		f.close()
		
