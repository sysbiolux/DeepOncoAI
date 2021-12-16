# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:05:33 2021

@author: sebde
"""
import numpy as np
import gseapy as gp
from gseapy.plot import barplot, dotplot
import matplotlib.pyplot as plt
import seaborn as sns

def get_enrichr(genelist, genesets, cutoff=None, tag=None):
    if tag is None:
        tag = ''
    if cutoff is None:
        cutoff = 0.5
    
    

    gene_list = genelist

    gene_sets = genesets

    enr = gp.enrichr(gene_list = gene_list,
                     gene_sets = gene_sets,
                     organism = 'Human',
                     description = tag,
                     outdir = 'enrichr',
                     # no_plot = True,
                     cutoff = cutoff
                    )
    f, ax = plt.subplots(1, 1, figsize=(15,15))
    title = tag
    barplot(enr.res2d, title=title )
    return enr

def get_gsea(genes, disease_genes, use_weights=False):

    incP = 1 / len(disease_genes)
    incN = -1 / (len(genes) - len(disease_genes))
    incList = np.repeat(incN, len(genes))
    disease_indices = [genes.index(x) for x in disease_genes]
    if not use_weights:
        incList[disease_indices] = incP
    else:
        incListData = -np.round( [len(genes)/(x+1) for x in range(len(genes))] , 0)
        temp = sum(incListData[disease_indices])
        incList[disease_indices] = incListData[disease_indices] / temp
    
    score = np.cumsum(incList)
    es = max(score)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 20))
    sns.lineplot(x = range(len(score)), y = score, ax = ax[0])
    
    randRuns = 20000
    esRand = []
    for counter in range(randRuns):
        temp = np.random.permutation(len(incList))
        incListRand = incList[temp]
        scoreListRand = np.cumsum(incListRand)
        esRand.append(max(scoreListRand))
    
    nes = es / np.mean(esRand)
    is_lower = es <= esRand
    pval = np.mean(is_lower.astype(int))
    print(pval)    
    sns.histplot(esRand, ax = ax[1])
    
    result = {'enrichment score': es, 'normalized es': nes, 'p-value': pval}
    
    return result