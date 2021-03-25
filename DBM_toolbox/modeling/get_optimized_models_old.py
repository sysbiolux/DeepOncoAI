# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:03:08 2020

@author: sebde
"""
import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier as RFC

from bayes_opt import BayesianOptimization


def get_optimized_models(data, targets, n_trials = 25):
	### SVM-RADIAL ############################################################
	
	def svc_cv(C, gamma, data, targets):
		"""SVC cross validation"""
		estimator = SVC(C=C, gamma=gamma, probability = True, random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_svc(data, targets, n_trials):
		"""Apply Bayesian Optimization to SVC parameters"""
		def svc_crossval(expC, expGamma):
			"""Wrapper of SVC cross validation.
			"""
			C = 10 ** expC
			gamma = 10 ** expGamma
			return svc_cv(C=C, gamma=gamma, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=svc_crossval,
			pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = SVC(
			C=10**(optimizer.max['params']['expC']), 
			gamma=10**(optimizer.max['params']['expGamma']), random_state=42)
		return optimizer.max, opt_model
	
	### SVM_LINEAR ############################################################
	
	def svml_cv(C, gamma, data, targets):
		"""SVM-linear cross validation.
		"""
		estimator = SVC(C=C, kernel= 'linear', gamma=gamma, probability = True, random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_svml(data, targets, n_trials):
		"""Apply Bayesian Optimization to SVC parameters."""
		def svml_crossval(expC, expGamma):
			"""Wrapper of SVC cross validation.
			"""
			C = 10 ** expC
			gamma = 10 ** expGamma
			return svml_cv(C=C, gamma=gamma, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=svml_crossval,
			pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = SVC(kernel='linear',
			C=10**(optimizer.max['params']['expC']), 
			gamma=10**(optimizer.max['params']['expGamma']), random_state=42)
		return optimizer.max, opt_model
	
	### SVM-POLYNOMIAL ###
	
	def svp_cv(C, gamma, data, targets):
		"""SVM-Polynomial cross validation.
		"""
		estimator = SVC(C=C, gamma=gamma, probability = True, kernel='poly', random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_svp(data, targets, n_trials):
		"""Apply Bayesian Optimization to SVC parameters."""
		def svp_crossval(expC, expGamma):
			"""Wrapper of SVC cross validation.
			"""
			C = 10 ** expC
			gamma = 10 ** expGamma
			return svp_cv(C=C, gamma=gamma, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=svp_crossval,
			pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = SVC(kernel='poly',
			C=10**(optimizer.max['params']['expC']), 
			gamma=10**(optimizer.max['params']['expGamma']), random_state=42)
		return optimizer.max, opt_model
		
	### RANDOM FORESTS ############################################################
	
	def rfc_cv(n_estimators, min_samples_split, max_features, max_depth, data, targets):
		"""Random Forest cross validation.
		"""
		estimator = RFC(
			n_estimators=n_estimators,
			min_samples_split=min_samples_split,
			max_features=max_features,
			max_depth=max_depth,
			random_state=42
		)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets,
			scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	
	def optimize_rfc(data, targets, n_trials):
		"""Apply Bayesian Optimization to Random Forest parameters."""
		def rfc_crossval(n_estimators, min_samples_split, max_features, max_depth):
			"""Wrapper of RandomForest cross validation.
			"""
			return rfc_cv(
				n_estimators=int(n_estimators),
				min_samples_split=int(min_samples_split),
				max_features=max(min(max_features, 0.999), 1e-3),
				max_depth=max_depth,
				data=data,
				targets=targets,
			)
		
		optimizer = BayesianOptimization(
			f=rfc_crossval,
			pbounds={
				"n_estimators": (10, 250),
				"min_samples_split": (2, 25),
				"max_features": (0.1, 0.999),
				"max_depth": (2, 50),
			},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = RFC(
			n_estimators=int(optimizer.max['params']['n_estimators']),
			min_samples_split=int(optimizer.max['params']['min_samples_split']),
			max_features=optimizer.max['params']['max_features'],
			max_depth=optimizer.max['params']['max_depth'],
			random_state=42
		)
		return optimizer.max, opt_model
	
	### LOGISTIC ############################################################
	def log_cv(C, tol, data, targets):
		"""Logistic cross validation."""
		estimator = LogisticRegression(C=C, tol=tol, random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_log(data, targets, n_trials):
		"""Apply Bayesian Optimization to Logistic parameters."""
		def log_crossval(expC, expTol):
			"""Wrapper of SVC cross validation.
			"""
			C = 10 ** expC
			tol = 10 ** expTol
			return log_cv(C=C, tol=tol, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=log_crossval,
			pbounds={"expC": (-3, 2), "expTol": (-5, -1)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = LogisticRegression(
			C=10**(optimizer.max['params']['expC']), 
			tol=10**(optimizer.max['params']['expTol']),
			random_state=42)
		return optimizer.max, opt_model
	
	### RIDGE ############################################################
	def ridge_cv(alpha, tol, data, targets):
		"""Ridge cross validation."""
		estimator = RidgeClassifier(alpha=alpha, tol=tol, random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_ridge(data, targets, n_trials):
		"""Apply Bayesian Optimization to ridge parameters."""
		def ridge_crossval(alpha, expTol):
			"""Wrapper of ridge cross validation.
			"""
			alpha = alpha
			tol = 10 ** expTol
			return ridge_cv(alpha = alpha, tol=tol, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=ridge_crossval,
			pbounds={"alpha": (0, 1000), "expTol": (-5, -1)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = RidgeClassifier(
			alpha=optimizer.max['params']['alpha'], 
			tol = 10 ** (optimizer.max['params']['expTol']), 
			random_state=42)
		return optimizer.max, opt_model
	
	### EXTRA-TREES ############################################################
	def et_cv(n_estimators, max_depth, data, targets):
		"""Logistic cross validation."""
		estimator = ExtraTreesClassifier(
			n_estimators = n_estimators,
			max_depth = max_depth,
			random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_et(data, targets, n_trials):
		"""Apply Bayesian Optimization to Extra-trees parameters."""
		def et_crossval(n_estimators, max_depth):
			"""Wrapper of Extra-trees cross validation.
			"""
			n_estimators = int(n_estimators)
			max_depth = int(max_depth)
			return et_cv(n_estimators=n_estimators, max_depth=max_depth, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=et_crossval,
			pbounds={"n_estimators": (10, 1000), "max_depth": (2, 50)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = ExtraTreesClassifier(
			n_estimators = int(optimizer.max['params']['n_estimators']),
			max_depth = int(optimizer.max['params']['max_depth']),
			random_state=42)
		return optimizer.max, opt_model
	
	### XGBOOST ############################################################
	def xgb_cv(max_depth, n_estimators, learning_rate, colsample_bytree, 
			 data, targets):
		"""XGBoost cross validation."""
		estimator = xgb.XGBClassifier(
			max_depth=max_depth,
			n_estimators=n_estimators,
			learning_rate=learning_rate,
			colsample_bytree=colsample_bytree,
			random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_xgb(data, targets, n_trials):
		"""Apply Bayesian Optimization to XGB parameters."""
		def xgb_crossval(max_depth, n_estimators, learning_rate, colsample_bytree):
			"""Wrapper of XGB cross validation.
			"""
			n_estimators = int(n_estimators)
			max_depth = int(max_depth)
			return xgb_cv(max_depth, n_estimators, learning_rate, colsample_bytree, 
			data, targets)
			
		optimizer = BayesianOptimization(
			f=xgb_crossval,
			pbounds={"max_depth": (10, 50),
			"n_estimators": (10, 1000),
			"learning_rate": (0.001, 0.05),
			"colsample_bytree": (0.2, 0.999),
			},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = xgb.XGBClassifier(
			max_depth=int(optimizer.max['params']['max_depth']),
			n_estimators=int(optimizer.max['params']['n_estimators']),
			learning_rate=optimizer.max['params']['learning_rate'],
			colsample_bytree=optimizer.max['params']['colsample_bytree'],
			random_state=42)
		return optimizer.max, opt_model
	
	### ADABOOST ############################################################
	def ada_cv(n_estimators, learning_rate, data, targets):
		"""AdaBoost cross validation."""
		estimator = AdaBoostClassifier(
			n_estimators=n_estimators, 
			learning_rate=learning_rate, 
			random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_ada(data, targets, n_trials):
		"""Apply Bayesian Optimization to AdaBoost parameters."""
		def ada_crossval(n_estimators, learning_rate):
			"""Wrapper of AdaBosst cross validation.
			"""
			n_estimators = int(n_estimators)
			return ada_cv(n_estimators=n_estimators, learning_rate=learning_rate,
					data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=ada_crossval,
			pbounds={"n_estimators": (10, 1000),
			"learning_rate": (0.001, 0.05)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = AdaBoostClassifier(
			n_estimators=int(optimizer.max['params']['n_estimators']), 
			learning_rate=optimizer.max['params']['learning_rate'], 
			random_state=42)
		return optimizer.max, opt_model
	
	### KNN #################################################################
	def knn_cv(k, data, targets):
		"""KNN cross validation."""
		estimator = KNeighborsClassifier(n_neighbors=k)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_knn(data, targets, n_trials):
		"""Apply Bayesian Optimization to KNN parameters."""
		def knn_crossval(k):
			"""Wrapper of KNN cross validation.
			"""
			k = int(k)
			return knn_cv(k=k, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=knn_crossval,
			pbounds={"k": (1, 100)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = KNeighborsClassifier(n_neighbors=int(optimizer.max['params']['k']))
		return optimizer.max, opt_model
	
	### GRADIENT BOOSTING ############################################################
	def gb_cv(learning_rate, n_estimators, subsample, max_depth,
			max_features, tol, data, targets):
		"""Gradient Boosting cross validation."""
		estimator = GradientBoostingClassifier(
			learning_rate=learning_rate,
			n_estimators=n_estimators,
			subsample=subsample,
			max_depth=max_depth,
			max_features=max_features,
			tol = tol,
			random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_gb(data, targets, n_trials):
		"""Apply Bayesian Optimization to Gradient Boosting parameters."""
		def gb_crossval(learning_rate, n_estimators, subsample, max_depth,
			max_features, expTol):
			"""Wrapper of Gradient Boosting cross validation.
			"""
			n_estimators = int(n_estimators)
			tol = 10 ** expTol
			return gb_cv(learning_rate=learning_rate,
				n_estimators=n_estimators,
				subsample=subsample,
				max_depth=max_depth,
				max_features=max_features,
				tol = tol,
				data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=gb_crossval,
			pbounds={"learning_rate": (0.001, 0.1),
			"n_estimators": (20, 1000),
			"subsample": (0.5, 0.999),
			"max_depth": (3, 20),
			"max_features": (0.2, 0.999),
			"expTol": (-4, 2),
			},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = GradientBoostingClassifier(
			learning_rate=optimizer.max['params']['learning_rate'],
			n_estimators=int(optimizer.max['params']['n_estimators']),
			subsample=optimizer.max['params']['subsample'],
			max_depth=optimizer.max['params']['max_depth'],
			max_features=optimizer.max['params']['max_features'],
			tol = 10 ** (optimizer.max['params']['expTol']),
			random_state=42)
		return optimizer.max, opt_model
	
	## PERCEPTRON ############################################################
	def mlp1_cv(nodes, alpha, data, targets):
		"""Perceptron validation."""
		estimator = MLPClassifier(
			hidden_layer_sizes=(nodes,), 
			alpha=alpha, 
			max_iter=1000, 
			random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_mlp1(data, targets, n_trials):
		"""Apply Bayesian Optimization to Perceptron parameters."""
		def mlp1_crossval(nodes, expAlpha):
			"""Wrapper of MLP cross validation.
			"""
			nodes = int(nodes)
			alpha= 10 ** expAlpha
			return mlp1_cv(nodes=nodes, alpha=alpha, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=mlp1_crossval,
			pbounds={"nodes": (5, 200), "expAlpha": (-6, -2)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = MLPClassifier(
			hidden_layer_sizes=(int(optimizer.max['params']['nodes']),), 
			alpha=10 ** (optimizer.max['params']['expAlpha']), 
			max_iter=1000, 
			random_state=42)
		return optimizer.max, opt_model
	
	def mlp2_cv(nodes1, nodes2, alpha, data, targets):
		"""Perceptron validation."""
		estimator = MLPClassifier(
			hidden_layer_sizes=(nodes1, nodes2), 
			alpha=alpha, 
			max_iter=1000, 
			random_state=42)
		cv = StratifiedKFold(n_splits=5)
		cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=cv, n_jobs=-1)
		return cval.mean()
	
	def optimize_mlp2(data, targets, n_trials):
		"""Apply Bayesian Optimization to Perceptron parameters."""
		def mlp2_crossval(nodes1, nodes2, expAlpha):
			"""Wrapper of MLP cross validation.
			"""
			nodes1 = int(nodes1)
			nodes2 = int(nodes2)
			alpha= 10 ** expAlpha
			return mlp2_cv(nodes1=nodes1, nodes2=nodes2,
					 alpha=alpha, data=data, targets=targets)
			
		optimizer = BayesianOptimization(
			f=mlp2_crossval,
			pbounds={"nodes1": (5, 200), "nodes2": (5, 200), "expAlpha": (-6, -2)},
			random_state=42,
			verbose=2
		)
		optimizer.maximize(n_iter= n_trials)
		
		print("Final result:", optimizer.max)
		opt_model = MLPClassifier(
			hidden_layer_sizes=(int(optimizer.max['params']['nodes1']), int(optimizer.max['params']['nodes2'])), 
			alpha= 10 ** (optimizer.max['params']['expAlpha']), 
			max_iter=1000, 
			random_state=42)
		return optimizer.max, opt_model
	
	
	Results = []
	print("--- Optimizing SVM-Radial ---")
	try:
		r_svc, best = optimize_svc(data, targets, n_trials)
		Results.append([r_svc, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing SVM-Linear ---")
	try:
		r_svml, best = optimize_svml(data, targets, n_trials)
		Results.append([r_svml, best])
	except:
		Results.append([0,0])
		
	print("--- Optimizing SVM-Polynomial ---")
	try:
		r_svp, best = optimize_svp(data, targets, n_trials)
		Results.append([r_svp, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing Random Forest ---")
	try:
		r_rfc, best = optimize_rfc(data, targets, n_trials)
		Results.append([r_rfc, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing Logistic ---")
	try:
		r_log, best = optimize_log(data, targets, n_trials)
		Results.append([r_log, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing Ridge ---")
	try:
		r_ridge, best = optimize_ridge(data, targets, n_trials)
		Results.append([r_ridge, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing Extra-trees ---")
	try:
		r_et, best = optimize_et(data, targets, n_trials)
		Results.append([r_et, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing XGBoost ---")
	try:
		r_xgb, best = optimize_xgb(data, targets, n_trials)
		Results.append([r_xgb, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing Adaboost ---")
	try:
		r_ada, best = optimize_ada(data, targets, n_trials)
		Results.append([r_ada, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing K-nearest neighbors ---")
	try:
		r_knn, best = optimize_knn(data, targets, n_trials)
		Results.append([r_knn, best])
	except:
		Results.append([0,0])
	
	print("--- Optimizing Gradient Boosting ---")
	try:
		r_gb, best = optimize_gb(data, targets, n_trials)
		Results.append([r_gb, best])
	except:
		Results.append([0,0])
	
	dataNN = (data*2)-1
	print("--- Optimizing Perceptron ---")
	try:
		r_mlp1, best = optimize_mlp1(dataNN, targets, n_trials)
		Results.append([r_mlp1, best])
	except:
		Results.append([0,0])
	try:
		r_mlp2, best = optimize_mlp2(dataNN, targets, n_trials)
		Results.append([r_mlp2, best])
	except:
		Results.append([0,0])
	
	
	list_models = ['svm', 'svl', 'svp', 'rf', 'log', 'ridge', 'et',
			'xgb', 'ada', 'knn', 'gbm', 'mlp1', 'mlp2', 'truth']
	cols = ['auc'] + [str(x) for x in range(targets.size)]
	summary = pd.DataFrame(index=list_models, columns = cols)
	list_opt_models = []
	list_trained_models = []
	for idx, model in enumerate(Results):
		print('predicting with model: ' + list_models[idx])
		if model[0]==0:
			print('---')
			summary.iloc[idx, 1:] = np.nan
		else:
			summary.iloc[idx, 0] = model[0]['target']
			opt_model = model[1]
			trained_model = opt_model.fit(data, targets)
			summary.iloc[idx, 1:] = cross_val_predict(opt_model, data, targets, cv=10, n_jobs=-1)
			list_opt_models.append(opt_model)
			list_trained_models.append(trained_model)
	return Results, summary, list_models, list_opt_models, list_trained_models
