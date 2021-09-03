# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:24:28 2020

@author: sebde
"""

def under_sample(df, y, strategy = 'random', final_ratio = 1):
    """performs undersampling of the majority class"""
    # TODO: implement this: 
    # 1) find majority and minority classes, find n and N

    if strategy=='random':
        #pick n random samples from majority
        df = df

    if strategy=='grid':
        #PCA majority instances
        #divide PCA space in grid
        #pick same number of majority instances in each grid cell
        df = df

    if strategy=='cluster':
        #cluster majority instances (k-means or dbscan)
        #count datapoints in each cluster d1, d2
        #pick a majority instance in each cluster in turn
        df = df

    return df

def over_sample(df, y, strategy = 'SMOTE', final_ratio = 1):
    """performs oversampling of the minority class"""
    # TODO: implement this:
    # 1) find majority and minority classes, find n and N

    if strategy=='SMOTE':
        df = df

    if strategy=='ADASYN':
        df = df

    if strategy=='random':
        df = df

    return df
