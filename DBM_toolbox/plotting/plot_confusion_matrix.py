# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:10:31 2020

@author: sebde
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes, ax=None, normalize=False, title=None, cmap=plt.cm.Oranges):
	"""This function prints and plots the confusion matrix.
	Normalization can be applied by setting 'normalize=True'"""
	if ax == None:
		fig, ax = plt.subplots()
	plt.tight_layout
	
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
		vmax=1
	else:
		print('Confusion matrix, without normalization')
		vmax=np.max(np.max(cm, axis=1))
	print(cm)
	
	ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
	#ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True', xlabel='Predicted')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	
	fmt = '.2f' if normalize else 'd'
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='black', weight= 'bold', fontsize=20)
	return ax
	
