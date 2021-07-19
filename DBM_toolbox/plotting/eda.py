# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:52:06 2020

@author: sebde
"""

import pandas as pd
import numpy as np
import seaborn as sns


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import missingno as msno

def plot_eda_all(dataframe):
	"""get plots for general data exploration"""
	ts = str(round(datetime.datetime.now().timestamp()))
	ncol = dataframe.shape[1]
	l = np.intc(np.ceil(np.sqrt(ncol)))
	c = np.intc(np.ceil(ncol/l))
		
	try:
		f, axes = plt.subplots(figsize=(20,50), sharex = True, sharey = True)
		zg = sns.violinplot(data=dataframe, ax=axes)
		zg.set_xticklabels(zg.get_xticklabels(), rotation=90)
		plt.savefig(ts + '_distrib.pdf')
	except:
		print('no plot 1')
	
	try:
		f, axes = plt.subplots(l, c, figsize=(20,50), sharex = True, sharey = True)
		axes = axes.ravel()
		
		for count, col in enumerate(dataframe.columns):
			dataframe2 = dataframe[col].dropna()
			zg = sns.distplot(dataframe2, kde=True, rug=True, ax=axes[count])
		#	zg.set_xlim(0,1)
			zg.set_title(col)
		plt.savefig(ts + '2.pdf')
	except:
		print('no plot 2')
	
	try:
		fig, ax = plt.subplots()
		cmap = sns.diverging_palette(220, 10, as_cmap=True)
		sns.heatmap(dataframe.corr(), vmin = 0, vmax = 1, cmap = cmap)
		plt.savefig(ts + 'correl.pdf')
	except:
		print('no plot 3')
	
	try:
		fig, ax = plt.subplots()
		means = dataframe.mean().rename('Mean')
		stds = dataframe.std().rename('Std')
		toplot = pd.concat([means, stds], axis = 1)
		sns.scatterplot(x = 'Mean', y = 'Std', data = toplot)
		plt.savefig(ts + '_mean-sd.pdf')
	except:
		print('no plot 4')

def plot_missing(dataframe, omic, database):
	ts = str(round(datetime.datetime.now().timestamp()))
	fig, ax = plt.subplots()
	msno.matrix(dataframe)
	plt.title(database + '_' + omic)
	plt.savefig(ts + '_missing.pdf')
# 	fig, ax = plt.subplots()
# 	msno.heatmap(dataframe)
# 	plt.title(database + '_' + omic)
# 	plt.savefig(ts + '_missing-correl.pdf')

def plot_results(dataframe):
	targets = list(set(dataframe['target']))
	for this_target in targets:
		ax = sns.barplot(x='algo', y='perf', hue='omic', data=dataframe).set_title('this_target')











