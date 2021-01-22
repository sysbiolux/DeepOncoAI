# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:52:06 2020

@author: sebde
"""

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_eda(df):
	"""get plots for general data exploration"""
	try:
		f, axes = plt.subplots(figsize=(20,50), sharex = True, sharey = True)
		zg = sns.violinplot(data=df, ax=axes)
		zg.set_xticklabels(zg.get_xticklabels(), rotation=45)
		
		ncol = df.shape[1]
		l = np.intc(np.ceil(np.sqrt(ncol)))
		c = np.intc(np.ceil(ncol/l))
	except:
		print('no plot')
	
	try:
		f, axes = plt.subplots(l, c, figsize=(20,50), sharex = True, sharey = True)
		
		axes = axes.ravel()
		count = 0
		
		for col in df.columns:
			df2 = df[col]
			df2 = df2.dropna()
			# TODO: this is broken, find a way to fit and show the best distribution:
			#bestDistrib, bestParams = best_fit_distribution(df2)
			#pdf = make_pdf(bestDistrib, bestParams)
			zg = sns.distplot(df2, kde=True, rug=True, label='data', ax=axes[count])
	#		zg.set_xlim(-1,1)
	#		zg.set_title(col)
	#		sns.lineplot(x=pdf.index, y=pdf, label=bestDistrib, ax=axes[count])
			count = count + 1
	except:
		print('no plot 2')
	
	try:
		fig, ax = plt.subplots()
		cmap = sns.diverging_palette(220, 10, as_cmap=True)
		sns.heatmap(df.corr(), vmin = -1, vmax = 1, cmap = cmap)
	except:
		print('no plot 3')
	
	try:
		fig, ax = plt.subplots()
		Means = df.mean().rename('Mean')
		Stds = df.std().rename('Std')
		toplot = pd.concat([Means, Stds], axis = 1)
		sns.scatterplot(x = 'Mean', y = 'Std', data = toplot)
	except:
		print('no plot 4')