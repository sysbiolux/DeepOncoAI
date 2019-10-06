# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""
import warnings
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns
sns.set(context='talk')


def get_classification_models(models=dict(), depth = 1):
	# linear models
	models['logistic'] = LogisticRegression()
	if depth == 1:
		alpha = [0.01, 0.2, 0.5, 0.8, 0.99, 1.0]
	else:
		alpha = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
	
	for a in alpha:
		models['ridge-'+str(a)] = RidgeClassifier(alpha=a)
	models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
	models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
	# non-linear models
	if depth == 1:
		n_neighbors = [1, 2, 3, 5, 10, 20, 50]
	else:
		n_neighbors = range(1, 100)
	
	for k in n_neighbors:
		models['knn-'+str(k)] = KNeighborsClassifier(n_neighbors=k)
	models['cart'] = DecisionTreeClassifier()
	models['extra'] = ExtraTreeClassifier()
	models['svml'] = SVC(kernel='linear')
	models['svmp'] = SVC(kernel='poly')
	if depth == 1:
		c_values = [0.01, 0.2, 0.5, 0.8, 0.99]
	c_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
	for c in c_values:
		models['svmr'+str(c)] = SVC(C=c)
	models['bayes'] = GaussianNB()
	# ensemble models
	n_trees = 100
	models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
	models['bag'] = BaggingClassifier(n_estimators=n_trees)
	models['rf'] = RandomForestClassifier(n_estimators=n_trees)
	models['et'] = ExtraTreesClassifier(n_estimators=n_trees)
	models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)
	models['xgb'] = xgb.XGBClassifier(n_estimators=n_trees, nthread=-1)
	
	print('Defined %d models' % len(models))
	return models
 




def get_regression_models(models=dict(), depth = 1):
	# linear models
	
	models['lr'] = LinearRegression()
	if depth == 1:
		alpha = [0.0, 0.01, 0.5, 2]
	else:
		alpha = [0.0, 0.01, 0.1, 0.2, 0.5, 0.7, 1, 2]

	for a in alpha:
		models['lasso-'+str(a)] = Lasso(alpha=a)
	for a in alpha:
		models['ridge-'+str(a)] = Ridge(alpha=a)
	for a1 in alpha:
		for a2 in alpha:
			name = 'en-' + str(a1) + '-' + str(a2)
			models[name] = ElasticNet(a1, a2)
	if depth > 1:
		
		models['huber'] = HuberRegressor()
		models['lars'] = Lars()
		models['llars'] = LassoLars()
		models['pa'] = PassiveAggressiveRegressor(max_iter=10000, tol=1e-4)
		models['ranscac'] = RANSACRegressor()
		models['sgd'] = SGDRegressor(max_iter=10000, tol=1e-4)
		models['theil'] = TheilSenRegressor(n_jobs=-1)
	
	# non-linear models
	if depth == 1:
		n_neighbors = [2, 5, 7, 20]
	else:
		n_neighbors = [2, 3, 4, 5, 7, 10, 20, 50]
	
	for k in n_neighbors:
		models['knn-'+str(k)] = KNeighborsRegressor(n_neighbors=k)
	models['cart'] = DecisionTreeRegressor()
	models['extra'] = ExtraTreeRegressor()
	models['svml'] = SVR(kernel='linear')
	models['svmp'] = SVR(kernel='poly')
	if depth == 1:
		c_values = [0.01, 0.2, 0.5, 0.75, 0.99, 1]
	else:
		c_values = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
		
	for c in c_values:
		models['svmr'+str(c)] = SVR(C=c)
	# ensemble models
	if depth == 1:
		n_trees = 200
	else:
		n_trees = 2000
	
	models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
	models['bag'] = BaggingRegressor(n_estimators=n_trees, n_jobs=-1)
	models['rf'] = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
	models['et'] = ExtraTreesRegressor(n_estimators=n_trees, n_jobs=-1)
	models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
	models['xgb'] = xgb.XGBRegressor(n_estimators=n_trees, nthread=-1)
	if depth == 1:
		n1_values = [1, 10, 100]
		n2_values = [1, 10, 100]
		n3_values = [1, 10, 100]
	else:
		n1_values = [1, 5, 10, 50, 200]
		n2_values = [1, 5, 10, 50, 200]
		n3_values = [1, 5, 10, 50, 200]
	
	for n1 in n1_values:
		for n2 in n2_values:
			for n3 in n3_values:
				models['mlp'+str(n1)+'-'+str(n2)+'-'+str(n3)] = MLPRegressor(solver='sgd', learning_rate_init=0.01, hidden_layer_sizes=(n1, n2, n3), verbose=False,  tol=0.00001, n_iter_no_change=1000, batch_size = 32, max_iter=100000)
	print('Defined %d models' % len(models))
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
#	steps.append(('standardize', StandardScaler()))
	# normalization
#	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# evaluate a single model
def evaluate_model(X, y, model, folds, metric):
	# create the pipeline
	pipeline = make_pipeline(model)
	# evaluate model
	scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)

	return scores

# evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, X_test, folds, metric):
	scores = None
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			scores = evaluate_model(X, y, model, folds, metric)
			fitModel = model.fit(X, y)
			predictions = model.predict(X_test)
	except:
		scores = None
		predictions = None
	return (scores, predictions)

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, X_test, folds=10, metric='accuracy'):
	results = dict()
	predicted = dict()
	for name, model in models.items():
		# evaluate the model
		scores, predictions = robust_evaluate_model(X, y, model, X_test, folds, metric)
		# show process
		if scores is not None:
			# store a result
			results[name] = scores
			predicted[name] = predictions
			mean_score, std_score, = mean(scores), std(scores)
			print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
		else:
			print('>%s: error' % name)
	return (results, predicted)

# print and plot the top n results
def summarize_results(results, predicted, y_test, thisCol, maximize=True, top_n=20):
	# check for no results
	if len(results) == 0:
		print('no results')
		return
	# determine how many results to summarize
	n = min(top_n, len(results))
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,mean(v)) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	# retrieve the top n for summarization
	names = [x[0] for x in mean_scores[:n]]
	scores = [results[x[0]] for x in mean_scores[:n]]
	predict = [predicted[x[0]] for x in mean_scores[:n]]
	# print the top n
	print()
	for i in range(n):
		name = names[i]
		mean_score, std_score = mean(results[name]), std(results[name])
		print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
	# boxplot for the top n
	plt.boxplot(scores, labels=names)
	_, labels = plt.xticks()
	plt.setp(labels, rotation=90)
	thisTitle = (thisCol+'_spotcheck.png')
	plt.savefig(thisTitle)
	f, axes = plt.subplots(4,5)
	for i in range(4):
		for j in range(5):
			axes[i,j].plot(y_test, predict[i*5+j], '.k')
			axes[i,j].set_title(names[i*5+j])
	f.suptitle(thisCol+'_Best_models_predictions')
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.savefig(thisCol+'_predictions.png')

	return scores


