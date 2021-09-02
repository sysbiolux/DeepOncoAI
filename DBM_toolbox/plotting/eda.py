# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:52:06 2020

@author: sebde
"""

from matplotlib import pyplot as plt
import matplotlib.colors as c
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import random

from DBM_toolbox.data_manipulation import preprocessing
# import missingno as msno

def doublesort(dataframe, ascending=True):
    dataframe['samplecompleteness'] = np.mean(dataframe, axis=1)
    dataframe = dataframe.append(pd.Series(data=np.mean(dataframe, axis=0), name='featurecompleteness'))
    semisorted_dataframe = dataframe.sort_values(by='samplecompleteness', ascending=True, axis=0)
    sorted_indices = semisorted_dataframe['samplecompleteness']
    semisorted_dataframe = semisorted_dataframe.drop('samplecompleteness', axis=1)
    sorted_dataframe = semisorted_dataframe.sort_values(by='featurecompleteness', ascending=True, axis=1)
    sorted_columns = sorted_dataframe.loc['featurecompleteness',:]
    sorted_dataframe = sorted_dataframe.drop('featurecompleteness')
    
    return sorted_dataframe, sorted_columns, sorted_indices
    

def plot_eda_all(dataframe, title=None):
    """get plots for general data exploration"""
    
    ts = str(round(datetime.datetime.now().timestamp()))
    
#     plot_eda_PCA(dataframe = dataframe, title=title, ts=ts)
    
    plot_eda_generaldistrib(dataframe = dataframe, title=title, ts=ts)
    
    plot_eda_meanvariance(dataframe = dataframe, title=title, ts=ts)
    
    plot_eda_missingsummary(dataframe = dataframe, title=title, ts=ts)
    
    plot_eda_correl(dataframe = dataframe, title=title, ts=ts)
    
    plot_eda_missingcorrel(dataframe = dataframe, title=title, ts=ts)
    
    
    
def plot_overlaps(dataframe):
    pass ##TODO: overlaps of datasets
    

def plot_eda_PCA(dataframe, title, ts):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(dataframe.dropna())
    principal_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
    
#     final_df = pd.concat([principal_df, df[['target']]], axis = 1)
    fig, ax = plt.subplots(figsize = (15,15))
    ax = sns.scatterplot(x='PC1', y='PC2', data=principal_df)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)



def plot_eda_generaldistrib(dataframe, title, ts):
    
    print(f'general distribution plot for {title}...')
    
    ncol = dataframe.shape[1]
    if ncol > 100:
        dataframe = dataframe.iloc[:, random.sample(range(dataframe.shape[1]), 100)]
        title = title + '_sample'
    try:
        fig, axes = plt.subplots(2, 1, figsize=(25,10), sharex=True)
        sns.set_context('talk')
        distr = sns.stripplot(data=dataframe, jitter=True, color='k', size=3, ax=axes[0])
        distr.set_xticklabels(distr.get_xticklabels(), rotation=90)
        distr.set_title('raw')
        distr2 = sns.stripplot(data=np.log10(dataframe), jitter=True, color='k', size=3, ax=axes[1])
        distr2.set_xticklabels(distr2.get_xticklabels(), rotation=90)
        distr2.set_title('log')
        fig.suptitle(title)
        
        plt.savefig(ts + '_' +  title + '_distrib.svg')
    except:
        print('no stripplot')
    
def plot_eda_meanvariance(dataframe, title, ts):
    
    print(f'mean-variance plot for {title}...')
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        sns.set_context('talk')
        means = dataframe.mean().rename('Mean')
        stds = dataframe.std().rename('Std')
        toplot = pd.concat([means, stds], axis = 1)
        mv = sns.scatterplot(x = 'Mean', y = 'Std', data = toplot, color='k', ax=axes[0])
        mv.set_title('raw')
        means = (np.log10(dataframe)).mean().rename('Mean')
        stds = (np.log10(dataframe)).std().rename('Std')
        toplot = pd.concat([means, stds], axis = 1).dropna()
        mv2 = sns.scatterplot(x = 'Mean', y = 'Std', data = toplot, color='k', ax=axes[1])
        mv2.set_title('log')
        fig.suptitle(title)
        
        plt.savefig(ts + '_' +  title + '_mean-sd.svg')
    except:
        print('no mean-variance plot')
    
def plot_eda_missingsummary(dataframe, title, ts):
    
    print(f'missing data plot for {title}...')
    
    try:
        bool_df = ~dataframe.isna()
        
        bool_df, sorted_featcompl, sorted_samplecompl = doublesort(bool_df)
        
        fig = plt.figure(figsize=(22, 22))
        sns.set_context('talk')
        gs = GridSpec(5, 5)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
#         ax3 = fig.add_subplot(gs[2:,1:])
        
        line1 = sns.lineplot(data=sorted_featcompl.to_numpy()[::-1], ax=ax1)
        line1.set_ylabel('feature completeness')
        
        line2 = sns.lineplot(data=sorted_samplecompl.to_numpy()[::-1], ax=ax2)
        line2.set_ylabel('sample completeness')
        
#         colors = {'black':1, 'white':0}
#         cMap = c.ListedColormap(colors)
#         miss = sns.heatmap(data=bool_df, cbar=False, cmap=cMap, ax=ax3)
#         fig.suptitle(title)
#         plt.savefig(ts + '_' +  title + '_missing.svg')
    except:
        print('no missing data plot')
    
    
def plot_eda_correl(dataframe, title, ts):
    
    print(f'correlation plot for {title}...')
    
    ncol = dataframe.shape[1]
    if ncol > 5000:
        dataframe = dataframe.iloc[:, random.sample(range(dataframe.shape[1]), 5000)]
        title = title + '_sample'
    
    try:
        featcorrel = dataframe.corr()
        sorted_featcorrel = np.abs(doublesort(featcorrel)[0])
        samplecorrel = dataframe.transpose().corr()
        sorted_samplecorrel = np.abs(doublesort(samplecorrel)[0])
        
        fig = plt.figure(figsize=(22, 22))
        sns.set_context('talk')
        gs=GridSpec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
#         ax3 = fig.add_subplot(gs[1:, 0])
#         ax4 = fig.add_subplot(gs[1:, 1])
        
        line1 = sns.distplot(sorted_featcorrel.to_numpy().flatten(), norm_hist=True, ax=ax1)
        line1.set_title('feature correlations')
        line2 = sns.distplot(sorted_samplecorrel.to_numpy().flatten(), norm_hist=True, ax=ax2)
        line2.set_title('sample correlations')
        
#         corr1 = sns.heatmap(data=sorted_featcorrel, cbar=False, cmap='magma_r', ax=ax3)
#         corr1.set_title('feature correlation matrix')
#         corr2 = sns.heatmap(data=sorted_samplecorrel, cbar=False, cmap='magma_r', ax=ax4)
#         corr2.set_title('sample correlation matrix')
#         fig.suptitle(title)
#         plt.savefig(ts + '_' +  title + '_correl.svg')
    except:
        print('no correlation plot')
        
def plot_eda_missingcorrel(dataframe, title, ts):
    
    print(f'missing data correlation plot for {title}...')
    
    ncol = dataframe.shape[1]
    if ncol > 5000:
        dataframe = dataframe.iloc[:, random.sample(range(dataframe.shape[1]), 5000)]
        title = title + '_sample'
    
    try:
        isdata = ~dataframe.isna()
        miss_featcorrel = isdata.astype(int).corr()
        sorted_missfeatcorrel = np.abs(doublesort(miss_featcorrel)[0])
        miss_samplecorrel = isdata.transpose().corr()
        sorted_misssamplecorrel = np.abs(doublesort(miss_samplecorrel)[0])
        
        fig = plt.figure(figsize=(22, 22))
        sns.set_context('talk')
        gs=GridSpec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
#         ax3 = fig.add_subplot(gs[1:, 0])
#         ax4 = fig.add_subplot(gs[1:, 1])
        
        line1 = sns.distplot(sorted_missfeatcorrel.to_numpy().flatten(), norm_hist=True, ax=ax1)
        line1.set_title('missing feature correlations')
        line2 = sns.distplot(sorted_misssamplecorrel.to_numpy().flatten(), norm_hist=True, ax=ax2)
        line2.set_title('missing sample correlations')
        
#         corr1 = sns.heatmap(data=sorted_missfeatcorrel, cbar=False, cmap='magma_r', ax=ax3)
#         corr1.set_title('missing feature correlation matrix')
#         corr2 = sns.heatmap(data=sorted_misssamplecorrel, cbar=False, cmap='magma_r', ax=ax4)
#         corr2.set_title('missing sample correlation matrix')
#         fig.suptitle(title)
#         plt.savefig(ts + '_' + title + '_missingcorrel.svg')
    except:
        print('no missing data correlation plot')
    








#     
#     
#     
#     
#     l = np.intc(np.ceil(np.sqrt(ncol)))
#     c = np.intc(np.ceil(ncol/l))
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     try:
#         f, axes = plt.subplots(figsize=(20,50), sharex = True, sharey = True)
#         zg = sns.violinplot(data=dataframe, ax=axes)
#         zg.set_xticklabels(zg.get_xticklabels(), rotation=90)
#         plt.savefig(ts + '_distrib.pdf')
#     except:
#         print('no plot 1')
#     
#     try:
#         f, axes = plt.subplots(l, c, figsize=(20,50), sharex = True, sharey = True)
#         axes = axes.ravel()
#         
#         for count, col in enumerate(dataframe.columns):
#             dataframe2 = dataframe[col].dropna()
#             zg = sns.distplot(dataframe2, kde=True, rug=True, ax=axes[count])
#         #    zg.set_xlim(0,1)
#             zg.set_title(col)
#         plt.savefig(ts + '2.pdf')
#     except:
#         print('no plot 2')
#     
#     try:
#         fig, ax = plt.subplots()
#         cmap = sns.diverging_palette(220, 10, as_cmap=True)
#         sns.heatmap(dataframe.corr(), vmin = 0, vmax = 1, cmap = cmap)
#         plt.savefig(ts + 'correl.pdf')
#     except:
#         print('no plot 3')
#     
#     try:
#         fig, ax = plt.subplots()
#         means = dataframe.mean().rename('Mean')
#         stds = dataframe.std().rename('Std')
#         toplot = pd.concat([means, stds], axis = 1)
#         sns.scatterplot(x = 'Mean', y = 'Std', data = toplot)
#         plt.savefig(ts + '_mean-sd.pdf')
#     except:
#         print('no plot 4')

# def plot_missing(dataframe, omic, database):
#     ts = str(round(datetime.datetime.now().timestamp()))
#     fig, ax = plt.subplots()
#     msno.matrix(dataframe)
#     plt.title(database + '_' + omic)
#     plt.savefig(ts + '_missing.pdf')
# #     fig, ax = plt.subplots()
# #     msno.heatmap(dataframe)
# #     plt.title(database + '_' + omic)
# #     plt.savefig(ts + '_missing-correl.pdf')

def plot_target(dataframe, bounds):
    title = dataframe.name
    ts = str(round(datetime.datetime.now().timestamp()))
    fig, ax = plt.subplots(2, 1, figsize=(15, 15))
    sns.distplot(dataframe, bins=50, rug=True, ax=ax[0])
    dataframe = preprocessing.rescale_data(dataframe)
    points = sns.kdeplot(dataframe, shade=True, ax=ax[1]).get_lines()[0].get_data()
    x = points[0]
    y = points[1]
    
    q = np.quantile(dataframe.dropna(), bounds)
    
    ax[1].fill_between(x,y, where = x >= q[1], color = 'g')
    ax[1].fill_between(x,y, where = x <= q[0], color = 'r')
    ax[1].fill_between(x,y, where = (x <= q[1]) & (x >= q[0]), color = 'y')
    fig.suptitle(title)
    plt.savefig(ts + '_' + title + '_distr.pdf')

def plot_results(dataframe):
    ts = str(round(datetime.datetime.now().timestamp()))
    targets = list(set(dataframe['target']))
    omics = list(set(dataframe['omic']))
    algos = list(set(dataframe['algo']))
    palette='colorblind'
    linewidth=1.5
    capsize=0.1
    edgecolor=".2"
    for this_target in targets:
        plt.figure(figsize=(15,15))
        ax = sns.barplot(x='algo', y='perf', hue='omic', 
                   palette=palette, linewidth=linewidth, capsize=capsize, edgecolor=edgecolor, ci=None, 
                   data=dataframe).set_title(this_target)
        plt.xticks(rotation=90)
        plt.savefig(ts + '_' + this_target + '_.pdf')
    for this_omic in omics:
        plt.figure(figsize=(15,15))
        ax = sns.barplot(x='target', y='perf', hue='algo', 
                   palette=palette, linewidth=linewidth, capsize=capsize, edgecolor=edgecolor, ci=None, 
                   data=dataframe).set_title(this_omic)
        plt.xticks(rotation=90)
        plt.savefig(ts + '_' + this_omic + '_.pdf')
    for this_algo in algos:
        plt.figure(figsize=(15,15))
        ax = sns.barplot(x='target', y='perf', hue='omic', 
                   palette=palette, linewidth=linewidth, capsize=capsize, edgecolor=edgecolor, ci=None, 
                   data=dataframe).set_title(this_algo)
        plt.xticks(rotation=90)
        plt.savefig(ts + '_' + this_algo + '_.pdf')
    for this_target in targets:
        plt.figure(figsize=(15,15))
        ax = sns.barplot(x='algo', y='perf', 
                   palette=palette, linewidth=linewidth, capsize=capsize, edgecolor=edgecolor, ci=None, 
                   data=dataframe).set_title(this_target)
        plt.xticks(rotation=90)
        plt.savefig(ts + '_' + this_target + '_2.pdf')
    for this_omic in omics:
        plt.figure(figsize=(15,15))
        ax = sns.barplot(x='target', y='perf', 
                   palette=palette, linewidth=linewidth, capsize=capsize, edgecolor=edgecolor, ci=None, 
                   data=dataframe).set_title(this_omic)
        plt.xticks(rotation=90)
        plt.savefig(ts + '_' + this_omic + '_.pdf')
    for this_algo in algos:
        plt.figure(figsize=(15,15))
        ax = sns.barplot(x='omic', y='perf',  
                   palette=palette, linewidth=linewidth, capsize=capsize, edgecolor=edgecolor, ci=None, 
                   data=dataframe).set_title(this_algo)
        plt.xticks(rotation=90)
        plt.savefig(ts + '_' + this_algo + '_.pdf')
    
    plt.figure(figsize=(15,15))
    ax = sns.scatterplot(x='perf', y='N', hue='target', style='algo', data=dataframe)
    plt.savefig(ts + '_' + '_overall_.pdf')











