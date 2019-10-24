# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:37:47 2019 by Sébastien De Landtsheer

"""
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes, ax=None, normalize=False, title=None, cmap=plt.cm.Oranges):
	from sklearn.metrics import confusion_matrix
	import numpy as np

	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	if ax == None:
		fig, ax = plt.subplots()
	plt.tight_layout
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='black')
	return ax

def plot_roc(y_test, X_test, y_train, X_train, thisNames, models, item=0, ax=None):
	from sklearn import metrics
	try:
		model = models[thisNames[item]]
		model.fit(X_train, y_train)
		fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
		auc = metrics.roc_auc_score(y_test,model.predict(X_test))
		if ax == None:
			fig, ax = plt.subplots()
		plt.tight_layout
		ax.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (thisNames[item], auc))
	except:
		print('The method %s does not support ROC curves' % thisNames[item])
	return ax

def plot_decision_boundary(y_test, X_test, y_train, X_train, thisNames, models, item=0, ax=None, pca=None):
	import pandas as pd
	import numpy as np
	from sklearn.decomposition import PCA
	if ax == None:
		fig, ax = plt.subplots()
	X = pd.concat([X_train, X_test])
	y = pd.concat([y_train, y_test])
	clf = models[thisNames[item]]
	clf.fit(X, y)
	pca = PCA(n_components = 2)
	pca = pca.fit(X)
	PCs = pca.transform(X)
	Xfake=pd.concat([X.copy(), X.copy(), X.copy()], ignore_index=True)
	for col in Xfake.columns:
		val = Xfake[col].values
		np.random.shuffle(val)
		Xfake[col] = val
		
	Z = clf.predict(Xfake)
	ZPCs = pca.transform(Xfake)
	missed = clf.predict(X_test) != y_test
	
	ax.scatter(ZPCs[:,0], ZPCs[:,1], c=Z, s=5, cmap='coolwarm', alpha=0.5, linewidths=0)
#	ax.scatter(PCs[:,0], PCs[:,1], c=y, s=20, cmap='bwr', edgecolor='k')
	ax.set_title('Decision boundary')

	plt.show()
	return ax



