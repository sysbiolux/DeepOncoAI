# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:00:28 2021

@author: sebde
"""
# Code from thomas:

# % enrichment score, to replace hypergeometric test?
# % todo:
# % - randomized sorting for equal reg load -> mean(nes)?
# % - non-expressed disease genes at end of sorted list?
# clc
# ​
# genes=1:10000; %full gene list
# temp=randperm(8000); %range for disease genes
# diseaseGenes=temp(1:100); %pick 100 disease genes in given disease gene range
# ​
# useWeights=0
# ​
# % calculate enrichment score
# incP=1/numel(diseaseGenes)
# incN=-1/(numel(genes)-numel(diseaseGenes))
# incList=ones(1,numel(genes))*incN;
# if useWeights~=1 %v1: equal step size (previous GSEA)
#     incList(diseaseGenes)=incP;
# else %use weights (GSEA PNAS 2005)
#     incListData=-round(numel(genes)./(1:numel(genes)));
#     temp=sum(incListData(diseaseGenes));
#     incList(diseaseGenes)=incListData(diseaseGenes)/temp;
# end
# ​
# scoreList=cumsum(incList);
# es=max(scoreList)
# figure, subplot(2,1,1)
# plot(scoreList)
# ​
# % randomize incList for background
# randRuns=2000;
# esRand=[];
# for counter=1:randRuns
#     temp=randperm(numel(incList));
#     incListRand=incList(temp);
#     scoreListRand=cumsum(incListRand);
#     esRand=[esRand, max(scoreListRand)];
# end
# nes=es/mean(esRand)
# pVal=mean(es<=esRand)
# subplot(2,1,2)
# hist(esRand,20)
# % figure
# % hist(esRand/mean(esRand),20)

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def gsea_num(genes, disease_genes, use_weights=False):
    incP = 1 / len(disease_genes)
    incN = -1 / (len(genes) - len(disease_genes))
    incList = np.repeat(incN, len(genes))
    if not use_weights:
        incList[disease_genes] = incP
    else:
        incListData = -np.round([len(genes) / x for x in genes], 0)
        temp = sum(incListData[disease_genes])
        incList[disease_genes] = incListData[disease_genes] / temp

    score = np.cumsum(incList)
    es = max(score)

    fig, ax = plt.subplots(2, 1, figsize=(15, 25))
    sns.lineplot(x=range(len(score)), y=score, ax=ax[0])

    randRuns = 2000
    esRand = []
    for counter in range(randRuns):
        temp = np.random.permutation(len(incList))
        incListRand = incList[temp]
        scoreListRand = np.cumsum(incListRand)
        esRand.append(max(scoreListRand))

    nes = es / np.mean(esRand)
    is_lower = list((es <= esRand).astype(int))
    pval = np.mean(is_lower)

    sns.histplot(esRand, ax=ax[1])

    print(pval)
    result = {"enrichment score": es, "normalized es": nes, "p-value": pval}

    return result


p = []
for rep in range(1000):
    genes = list(range(1, 10001))
    temp = np.random.permutation(9999) + 1
    disease_genes = temp[:100]
    y = gsea_num(genes, disease_genes, use_weights=True)
    p.append(y["p-value"])
fig, ax = plt.subplots(1, 1)
sns.histplot(p, ax=ax)


#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def gsea(genes, disease_genes, use_weights=False):

    incP = 1 / len(disease_genes)
    incN = -1 / (len(genes) - len(disease_genes))
    incList = np.repeat(incN, len(genes))
    disease_indices = [genes.index(x) for x in disease_genes]
    if not use_weights:
        incList[disease_indices] = incP
    else:
        incListData = -np.round([len(genes) / (x + 1) for x in range(len(genes))], 0)
        temp = sum(incListData[disease_indices])
        incList[disease_indices] = incListData[disease_indices] / temp

    score = np.cumsum(incList)
    es = max(score)

    fig, ax = plt.subplots(2, 1, figsize=(10, 20))
    sns.lineplot(x=range(len(score)), y=score, ax=ax[0])

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
    sns.histplot(esRand, ax=ax[1])

    result = {"enrichment score": es, "normalized es": nes, "p-value": pval}

    return result


genes = ["GENE" + str(x) for x in range(10000)]
temp = np.random.permutation(9999) + 1
disease_genes = [genes[x] for x in temp[:100]]

yaya = gsea(genes, disease_genes, use_weights=False)
