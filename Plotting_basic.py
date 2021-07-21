# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:43:53 2021

@author: apurva.badkas
"""

import logging
import random
import numpy as np
import datetime
import missingno as msno
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(filename='run.log', level=logging.INFO, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
from config import Config
from DBM_toolbox.data_manipulation import dataset_class
config = Config()

logging.info("Reading data")
data = config.read_data()

far = data
omics = far.omic
databases = far.database
for database in databases[0:1]:
    for omic in omics[1:2]:
        #logging.info(f"plotting info for {omic} in {database}")
        dataframe = far.to_pandas(omic=omic, database=database)
                
        if len(dataframe.columns) <= 100:
           eda.plot_eda_all(dataframe)
        else:
           pick = random.sample(range(dataframe.shape[1]), 100)
          # eda.plot_eda_all(dataframe.iloc[:, pick])
        #eda.plot_missing(dataframe, omic, database)
test = dataframe.iloc[:, pick]

import datetime
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


ts = str(round(datetime.datetime.now().timestamp()))
ncol = test.shape[1]
l = np.intc(np.ceil(np.sqrt(ncol)))
c = np.intc(np.ceil(ncol/l))
f, axes = plt.subplots(figsize=(20,50), sharex = True, sharey = True)
zg = sns.violinplot(data=test, ax=axes)
zg.set_xticklabels(zg.get_xticklabels(), rotation=90)
plt.savefig(ts + '_distrib.pdf')





#Organizing data

eig_vect = dataframe.loc[:, dataframe.columns.str.contains('eig')]
eig_vect = dataframe.loc[:, dataframe.columns.str.contains('bet')]
Miss_nos = pd.DataFrame(eig_vect.isna().sum(axis = 1))
eig_vect['Total_Nans'] = Miss_nos
eig_vect['Total_Feat'] = eig_vect.shape[1]
eig_vect['Tot_present_data'] = eig_vect['Total_Feat'] - eig_vect['Total_Nans']
eig_vect['Samples'] = eig_vect.index
eig_vect = eig_vect.sort_values(by = 'Tot_present_data', ascending=False)
# creating subplots
fig, ax = plt.subplots()
#ax = plt.subplots()
index_c = pd.DataFrame(eig_vect.index, columns = ['Samples'])  
# plotting columns
sns.barplot(y='Total_Feat', x='Samples', data = eig_vect, color='b',ax = ax)
sns.barplot(y='Total_Nans', x='Samples', data = eig_vect, color='r', ax=ax)
  

# renaming the axes
ax.set(xlabel="Samples", ylabel="Data present/Total data")
ax.set(xticklabels = [])
#ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
  
# visulaizing illustration
plt.show()


max_miss_feat = eig_vect.drop(columns = eig_vect.iloc[:,-4:-1])
max_miss_feat = max_miss_feat.T
max_miss_feat = pd.DataFrame(max_miss_feat.isna().sum(axis = 1), columns = ['Features'])
max_miss_feat = max_miss_feat.sort_values(by = 'Features')

fig, ax = plt.subplots()
#ax = plt.subplots()
# plotting columns
sns.barplot(y='Features', x = max_miss_feat.index, data = max_miss_feat, color='b')
  
# renaming the axes
ax.set(xlabel="Samples", ylabel="Data missing/Total data")
ax.set(xticklabels = [])
#ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
  
# visulaizing illustration
plt.show()


<<<<<<< Updated upstream
index_to_plot = max_miss_feat.iloc[0:50,:]
Data_to_plot = dataframe.loc[:,dataframe.columns.isin(index_to_plot.index)]

Cols = Data_to_plot.columns
fig, ax = plt.subplots()

ax = sns.boxplot(data=Data_to_plot, palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)



import numpy as np

ncol = Data_to_plot.shape[1]
l = np.intc(np.ceil(np.sqrt(ncol)))
c = np.intc(np.ceil(ncol/l))

f, axes = plt.subplots(l, c, figsize=(20,50), sharex = True, sharey = True)
axes = axes.ravel()
        
for count, col in enumerate(Data_to_plot.columns):
    dataframe2 = Data_to_plot[col].dropna()
    zg = sns.distplot(dataframe2, kde=True, rug=True, ax=axes[count])
    plt.xlim(0, 0.01)
    plt.ylim(0, 800)
        #    zg.set_xlim(0,1)
    zg.set_title(col)
plt.savefig('Dist2.pdf')plt.savefig('Dist2.pdf')plt.savefig('Dist2.pdf')plt.savefig('Dist2.pdf')
pick = random.sample(range(dataframe.shape[1]), 100)
                #    print(dataframe.iloc[:, pick])
 					


for i in pick:
    ts = str(round(datetime.datetime.now().timestamp()))
    
    index_to_plot = max_miss_feat.iloc[i:i+50,:]
    Data_to_plot = dataframe.loc[:,dataframe.columns.isin(index_to_plot.index)]
    
    Cols = Data_to_plot.columns
    fig, ax = plt.subplots()
    
    ax = sns.boxplot(data=Data_to_plot, palette="Set2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.savefig(ts + 'box.png')
    
    ncol = Data_to_plot.shape[1]
    l = np.intc(np.ceil(np.sqrt(ncol)))
    c = np.intc(np.ceil(ncol/l))

    
    
    f, axes = plt.subplots(l, c, figsize=(20,50), sharex = True, sharey = True)
    axes = axes.ravel()
            
    for count, col in enumerate(Data_to_plot.columns):
        dataframe2 = Data_to_plot[col].dropna()
        var = Data_to_plot[col].max()
        zg = sns.distplot(dataframe2, kde=True, rug=True, ax=axes[count])
        plt.xlim(0, 0.001)
        plt.ylim(0, 3000)
            #    zg.set_xlim(0,1)
        zg.set_title(col)
    plt.savefig(ts + '2.pdf')
>>>>>>> Stashed changes
