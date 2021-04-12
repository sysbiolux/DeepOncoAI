# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:52:06 2020

@author: sebde
"""

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime

def plot_eda(df):
	"""get plots for general data exploration"""
	ts = datetime.datetime.now().timestamp()
	ncol = df.shape[1]
	l = np.intc(np.ceil(np.sqrt(ncol)))
	c = np.intc(np.ceil(ncol/l))
		
	try:
		f, axes = plt.subplots(figsize=(20,50), sharex = True, sharey = True)
		zg = sns.violinplot(data=df, ax=axes)
		zg.set_xticklabels(zg.get_xticklabels(), rotation=90)
		plt.savefig([ts+'.pdf'])
	except:
		print('no plot 1')
	
	try:
		f, axes = plt.subplots(l, c, figsize=(20,50), sharex = True, sharey = True)
		axes = axes.ravel()
		
		for count, col in enumerate(df.columns):
			df2 = df[col].dropna()
			zg = sns.distplot(df2, kde=True, rug=True, ax=axes[count])
			zg.set_xlim(0,1)
			zg.set_title(col)
		plt.savefig([ts+'.pdf'])
	except:
		print('no plot 2')
	
	try:
		fig, ax = plt.subplots()
		cmap = sns.diverging_palette(220, 10, as_cmap=True)
		sns.heatmap(df.corr(), vmin = 0, vmax = 1, cmap = cmap)
		plt.savefig([ts+'.pdf'])
	except:
		print('no plot 3')
	
	try:
		fig, ax = plt.subplots()
		Means = df.mean().rename('Mean')
		Stds = df.std().rename('Std')
		toplot = pd.concat([Means, Stds], axis = 1)
		sns.scatterplot(x = 'Mean', y = 'Std', data = toplot)
		plt.savefig([ts+'.pdf'])
	except:
		print('no plot 4')