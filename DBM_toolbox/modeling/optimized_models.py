import logging

import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb

from bayes_opt import BayesianOptimization


class ParameterBound:

	def __init__(self, minimum, maximum, discrete=False, logarithmic=False):
		self.minimum = minimum
		self.maximum = maximum
		self.discrete = discrete
		self.logarithmic = logarithmic
		self.discrete = discrete

	def transform_bound(self):
		return self.transform(self.minimum), self.transform(self.maximum)

	def transform(self, value):
		if not self.logarithmic:
			return value
		return np.log(value)

	def inverse_transform(self, value):
		if not self.logarithmic:
			return value
		return np.exp(value)


def create_SVC(**kwargs):
	return SVC(probability=True, random_state=42, **kwargs)

def create_RFC(**kwargs):
	return RFC(random_state=42, n_jobs=-1, **kwargs)

def create_SVM(**kwargs):
	return SVC(kernel='linear', probability=True, random_state=42, **kwargs)

def create_SVP(**kwargs):
	return SVC(kernel='poly', probability=True, random_state=42, **kwargs)
	
def create_Logistic(**kwargs):
	return LogisticRegression(random_state=42, **kwargs)

def create_Ridge(**kwargs):
	return RidgeClassifier(random_state=42, **kwargs)

def create_ET(**kwargs):
	return ExtraTreesClassifier(random_state=42, n_jobs=-1, **kwargs)

def create_KNN(**kwargs):
	return KNeighborsClassifier(n_jobs=-1, **kwargs)

def create_XGB(**kwargs):
	return xgb.XGBClassifier(random_state=42, n_jobs=-1, **kwargs)

def create_Ada(**kwargs):
	return AdaBoostClassifier(random_state=42, **kwargs)

def create_GBM(**kwargs):
	return GradientBoostingClassifier(random_state=42, **kwargs)

def create_MLP1(**kwargs):
	return MLPClassifier(random_state=42, **kwargs)

def create_MLP2(**kwargs):
	return MLPClassifier(random_state=42, **kwargs)


models = [
	{'name': 'SVC', 'estimator_method': create_SVC, 'parameter_bounds': {
		'C': ParameterBound(10e-3, 10e2, logarithmic=True), 
		'gamma': ParameterBound(10e-4, 10e-1, logarithmic=True)}},
	{'name': 'RFC', 'estimator_method': create_RFC, 'parameter_bounds': {
		'n_estimators': ParameterBound(10, 250, discrete=True), 
		'min_samples_split': ParameterBound(2, 25, discrete=True),
		'max_features': ParameterBound(0.1, 0.999), 
		'max_depth': ParameterBound(2, 50, discrete=True)}},
	{'name': 'SVM', 'estimator_method': create_SVM, 'parameter_bounds': {
		'C': ParameterBound(10e-3, 10e2, logarithmic=True), 
		'gamma': ParameterBound(10e-4, 10e-1, logarithmic=True)}},
	{'name': 'SVP', 'estimator_method': create_SVP, 'parameter_bounds': {
		'C': ParameterBound(10e-3, 10e2, logarithmic=True), 
		'gamma': ParameterBound(10e-4, 10e-1, logarithmic=True)}},
	 {'name': 'Logistic', 'estimator_method': create_Logistic, 'parameter_bounds': {
	 	'C': ParameterBound(10e-1, 2, logarithmic=True), 
	 	'tol': ParameterBound(10e-5, 10e-1, logarithmic=True)}},
	 {'name': 'Ridge', 'estimator_method': create_Ridge, 'parameter_bounds': {
	 	'alpha': ParameterBound(0.1, 1000), 
	 	'tol': ParameterBound(10e-5, 10e-1, logarithmic=True)}},
	 {'name': 'ET', 'estimator_method': create_ET, 'parameter_bounds': {
	 	'n_estimators': ParameterBound(10, 1000, discrete=True), 
	 	'max_depth': ParameterBound(2, 50)}},
	 {'name': 'KNN', 'estimator_method': create_KNN, 'parameter_bounds': {
	 	'n_neighbors': ParameterBound(1,100, discrete=True)}},
	 {'name': 'XGB', 'estimator_method': create_XGB, 'parameter_bounds': {
	 	'max_depth' : ParameterBound(10, 50, discrete=True), 
	 	'n_estimators' : ParameterBound(10, 1000, discrete=True), 
	 	'learning_rate' : ParameterBound(0.001, 0.05), 
	 	'colsamples_bytree' : ParameterBound(0.2, 0.99)}},
	 {'name': 'Ada', 'estimator_method': create_Ada, 'parameter_bounds': {
	 	'n_estimators' : ParameterBound(10, 1000, discrete=True), 
	 	'learning_rate' : ParameterBound(0.001, 0.05)}},
	 {'name': 'GBM', 'estimator_method': create_GBM, 'parameter_bounds': {
	 	'learning_rate': ParameterBound(0.001, 0.1), 
	 	'n_estimators': ParameterBound(20, 1000, discrete=True), 
	 	'subsample': ParameterBound(0.5, 0.999), 
	 	'max_depth': ParameterBound(3, 20, discrete=True), 
	 	'max_features': ParameterBound(0.2, 0.999), 
	 	'tol': ParameterBound(10e-4, 10e2, logarithmic=True)}},
	 {'name': 'MLP1', 'estimator_method': create_MLP1, 'parameter_bounds': {
	 	'hidden_layer_sizes': ParameterBound(5, 200, discrete=True), 
	 	'alpha': ParameterBound(10e-6, 10e-2, logarithmic=True)}},
	 {'name': 'MLP2', 'estimator_method': create_MLP2, 'parameter_bounds': { # TODO: this needs to return a tuple, not a single value?
	 	'hidden_layer_sizes': ParameterBound(5, 200, discrete=True), 
	 	'alpha': ParameterBound(10e-6, 10e-2, logarithmic=True)}}
	]

def get_estimator_list():
	estimator_list = list()
	for model in models:
		estimator = model['estimator_method'].split('_')[-1]
		estimator_list.append(estimator)
	return estimator_list

def cross_validate_evaluation(estimator, data, targets, metric):
	cv = StratifiedKFold(n_splits=5)
	cval = cross_val_score(estimator, data, targets,
scoring=metric, cv=cv, n_jobs=-1)
	return cval.mean()

def create_pbounds_argument(parameter_bounds):
	pbounds = dict()
	for key in parameter_bounds.keys():
		pbounds[key] = parameter_bounds[key].transform_bound()
	return pbounds

def retrieve_original_parameters(optimizer_parameters, parameter_bounds):
	original_parameters = dict()
	for key in parameter_bounds.keys():
		original_parameters[key] = parameter_bounds[key].inverse_transform(optimizer_parameters[key])
	return original_parameters

def bayes_optimize_estimator(estimator_method, parameter_bounds, data,
targets, n_trials, metric):
	def instantiate_cross_validate_evaluation(**kwargs):
		for parameter_bound_name in parameter_bounds:
			parameter_bound = parameter_bounds[parameter_bound_name]
			if parameter_bound.discrete:
				kwargs[parameter_bound_name] = int(round(kwargs[parameter_bound_name]))
			if parameter_bound.logarithmic:
				kwargs[parameter_bound_name] = parameter_bound.inverse_transform(kwargs[parameter_bound_name])
		estimator = estimator_method(**kwargs)
		return cross_validate_evaluation(estimator, data, targets, metric)
	
	pbounds = create_pbounds_argument(parameter_bounds)
	
	optimizer = BayesianOptimization(
		f=instantiate_cross_validate_evaluation,
		pbounds=pbounds,
		random_state=42,
		verbose=2
	)
	optimizer.maximize(n_iter=n_trials)
	logging.info(f"Final result: {optimizer.max}")
	optimizer_parameters = optimizer.max['params']
	original_parameters = retrieve_original_parameters(optimizer_parameters, parameter_bounds)
	opt_model = estimator_method(**original_parameters)
	return optimizer.max, opt_model


def bayes_optimize_models(data, targets, n_trials=None, algos=None, metric=None):
	if n_trials is None:
		n_trials = 20
	if metric is None:
		metric = 'roc_auc'
	if algos is None:
		models_to_optimize = models
	else:
		models_to_optimize = []
		for model in models:
			if model['name'] in algos:
				models_to_optimize.append(model)
	optimal_models = dict()
	for model in models_to_optimize:
		logging.info(f"Bayes optimizing model {model['name']} for {targets.name}")
		try:
			maximum_value, optimal_model = bayes_optimize_estimator(model['estimator_method'],
																model['parameter_bounds'],
																data,
																targets,
																n_trials, metric)
		except:
			logging.info(f"Could not optimize {model['name']}")
			maximum_value = np.nan
			optimal_model = np.nan
			
		optimal_models[model['name']] = {'estimator': optimal_model, 'result': maximum_value}
	return optimal_models