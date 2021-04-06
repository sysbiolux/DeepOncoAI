# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:01:57 2020

@author: sebde
"""
import numpy as np
from vecstack import stacking
from sklearn.metrics import roc_auc_score

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