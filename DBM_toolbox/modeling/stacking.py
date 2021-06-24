# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:01:57 2020

@author: sebde
"""
import numpy as np
import pandas as pd
from vecstack import stacking
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

def stack_models(models, final_model, X_train, y_train, X_test, y_test, folds, metric=roc_auc_score, seed=42):
	"""Constructs, fits, predicts using XGBoost on a bag of models,
	using only the models' predictions as input
	"""
	print('stacking models...')
	S_train, S_test = stacking(models, X_train, y_train, X_test, 
							regression=False, 
							mode='oof_pred_bag',
							needs_proba=False,
							save_dir=None,
							metric=metric,
							n_folds=folds,
							stratified=True,
							shuffle=True,
							random_state=seed,
							verbose=0)
	model = final_model
	print('fitting stack...')
	model = model.fit(S_train, y_train)
	y_pred_train = model.predict(S_train)
	y_pred = model.predict(S_test)
	y_proba = model.predict_proba(S_test)
	train_AUC = roc_auc_score(y_train, y_pred_train)
	finalAUC = roc_auc_score(y_test, y_pred)
# 	print('Final AUC: [%.8f]' % finalAUC)
	return model, finalAUC, train_AUC, y_pred, y_proba


def stack_extended_models(models, final_model, X_train, y_train, X_test, y_test, folds, metric=roc_auc_score, seed=42):
	"""Constructs, fits, predicts using XGBoost on a bag of models,
	using all the training data plus the other models' predictions as input
	"""
	S_train, S_test = stacking(models, X_train, y_train, X_test,
							regression=False,
							mode='oof_pred_bag',
							needs_proba=False,
							save_dir=None,
							metric=metric,
							n_folds=folds,
							stratified=True,
							shuffle=True,
							random_state=seed,
							verbose=0)
	model = final_model
	E_train = np.concatenate([S_train, X_train],axis=1)
	E_test = np.concatenate([S_test, X_test], axis=1)
	model = model.fit(E_train, y_train)
	y_pred = model.predict(E_test)
	y_proba = model.predict_proba(E_test)
	finalAUC = roc_auc_score(y_test, y_pred)
	print('Final AUC: [%.8f]' % finalAUC)
	return model, finalAUC, y_pred, y_proba

def compute_stacks(dataset, models, final_model, targets_list, metric='roc_auc', folds=10, seed=42):
	best_stacks = dict()
	for target in targets_list:
		print(f'Computing stack for {target}')
		this_dataset = dataset.to_binary(target=target)
		y = this_dataset.to_pandas()[target]
		omics = models[target]
		predictions = pd.DataFrame(index = y.index)
		scores = pd.Series(index = ['full', 'lean'])
		importances = pd.DataFrame()
		for omic in omics.keys():
			if omic == 'complete':
				X = this_dataset.to_pandas().drop(targets_list, axis=1)
			else:
				X = this_dataset.to_pandas(omic=omic)
				index1 = y.index[y.apply(np.isnan)]  ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost
				index2 = X.index[X.apply(np.isnan).any(axis=1)]  ## SOLVED?
				indices_to_drop = index1.union(index2)
				
				X = X.drop(indices_to_drop)
				y = y.drop(indices_to_drop)
			
			
			models_list = omics[omic]
			for id, model in enumerate(models_list):
				
				this_model = model.iloc[0]
			
				xval = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
				print(X, y, xval)
				omic_predict = cross_val_predict(this_model, X, y, cv=xval, n_jobs=-1)
				feature_name = omic + str(id)
				predictions[feature_name] = omic_predict
		
		data_with_predictions = this_dataset.to_pandas().drop(targets_list, axis=1).merge(predictions, left_index=True, right_index=True)
		full_stack = final_model.fit(data_with_predictions, y, eval_metric='auc')
		full_stack_predict = cross_val_predict(final_model, data_with_predictions, y, cv=xval, n_jobs=-1)
		scores['full'] = np.mean(cross_val_score(final_model, data_with_predictions, y, scoring=metric, cv=xval, n_jobs=-1))
		
		lean_stack = final_model.fit(predictions, y, eval_metric='auc')
		lean_stack_predict = cross_val_predict(final_model, predictions, y, cv=xval, n_jobs=-1)
		scores['lean'] = np.mean(cross_val_score(final_model, predictions, y, scoring=metric, cv=xval, n_jobs=-1))
		
		predictions['full_stack'] = full_stack_predict
		predictions['lean_stack'] = lean_stack_predict
		predictions['truth'] = y.values
		
# 			rfecv = RFECV(estimator=final_model, step=1, cv=xval, scoring=metric, min_features_to_select=min_models)
# 			rfecv.fit(predictions, y)
# 			print("Optimal number of features : %d" % rfecv.n_features_)
# 			selected_features = rfecv.get_support()
# 			score = rfecv.score(X, y)
		try:
			importances['full'] = pd.DataFrame(full_stack.feature_importances_, index=full_stack.feature_names)
			importances['lean'] = pd.DataFrame(lean_stack.feature_importances_, index=lean_stack.feature_names)
		except:
			importances = np.nan
		best_stacks[target] = {'support': importances, 
								'scores': scores,
								'predictions': predictions}
	return best_stacks