# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:55:25 2020

@author: sebde
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def compare_train_test(training_data_filename, test_data_filename, test_id_string, target_string):
	"""returns plots for each variable to check the distributions in train and test sets and the 
	distributions of targets in missing data (bias in missing samples)"""
	a = pd.read_csv(training_data_filename)
	b = pd.read_csv(test_data_filename)
#	test_id = b[test_id_string]
	na = a.shape[0]
#	nb = b.shape[0]
	c = pd.concat((a, b), sort=False).reset_index(drop=True)
	target = a[target_string].to_frame()
	c.drop([target_string], axis=1, inplace=True)
	c['Dataset']='Test'
	c.iloc[0:na, c.shape[1]-1]='Train'
	
	cols = a.columns[0:-1]
	for t in cols:
		print(t)
		anul = a[t].isna()
		bnul = b[t].isna()
		print(anul.sum())
		print(bnul.sum())
		f, ax = plt.subplots(figsize=(15,15))
		plt.suptitle(t, fontsize=16)
		plt.subplot(2,2,1)
		if a[t].dtype != 'object':
			sns.distplot(a[t][anul==False])
			sns.distplot(b[t][bnul==False])
			plt.title('Dist')
			plt.legend(['Train', 'Test'])
			plt.subplot(2,2,2)
			sns.scatterplot(y=target_string, x=t, data=a)
			plt.title('Correl with target')
			if anul.sum() > 0:
				print(anul.sum())
				v1 = target[anul]
				v2 = target[anul==False]
				plt.subplot(2,2,3)
				sns.distplot(v1)
				sns.distplot(v2)
				plt.title('Target dist in Train')
				plt.legend(['Missing', 'Present'])
		else:
			sns.countplot(data=c, hue='Dataset', x=t)
			plt.title('Dist')
			plt.legend(['Train', 'Test'])
			plt.subplot(2,2,2)
			sns.boxplot(data = a, x=t, y='SalePrice')
			plt.title('Correl with target')
			if anul.sum() > 0:
				print(anul.sum())
				thistarget = target.copy()
				thistarget['Data'] = 'Present'
				thistarget.loc[a[t].isna(), 'Data'] = 'Missing'
				plt.subplot(2,2,3)
				sns.boxplot(data=thistarget, x='Data', y=target_string)
				plt.title('Target dist in Train')
				plt.legend(['Missing', 'Present'])
	
