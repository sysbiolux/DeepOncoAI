# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:05:33 2021

@author: sebde
"""
import gseapy as gp
from gseapy.plot import barplot, dotplot
import matplotlib.pyplot as plt

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

