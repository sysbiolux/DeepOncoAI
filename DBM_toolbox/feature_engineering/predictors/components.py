# -*- coding: utf-8 -*-
"""

"""
from DBM_toolbox.data_manipulation import dataset_class

import pandas as pd
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection


def make_dataset(dataframe, omic: str = None, database: str = None):
    dataset = dataset_class.Dataset(dataframe=dataframe, omic=omic, database=database)
    return dataset


def get_PCs(df, n_components: int = None, label: str = None):
    """returns the PCs columns corresponding to the PCA components of the dataframe"""
    if label is None:
        label = (datetime.now()).strftime("%Y%m%d%H%M%S")
    if n_components is None:
        n_components = 2
    pca = PCA(n_components=n_components)
    df = df.dropna()
    principal_components = pca.fit_transform(df)
    column_names = []
    for n in range(1, n_components + 1):
        column_names.append("PC" + str(n) + label)
    df_PCs = pd.DataFrame(
        data=principal_components, index=df.index, columns=column_names
    )
    var = pca.explained_variance_ratio_
    #     comp = pca.components_
    print("fraction of variance explained: %.5f (PC1), %.5f (PC2)" % (var[0], var[1]))
    return make_dataset(df_PCs, omic="PC", database="ENGINEERED")


def get_ICs(df, n_components: int = None, label: str = None, random_state=42):
    """returns the PCs columns corresponding to the ICA components of the dataframe"""
    if label is None:
        label = (datetime.now()).strftime("%Y%m%d%H%M%S")
    if n_components is None:
        n_components = 2
    transformer = FastICA(n_components=n_components, random_state=random_state)
    X_transformed = transformer.fit_transform(df)
    column_names = []
    for n in range(1, n_components + 1):
        column_names.append("IC" + str(n) + label)
    df_ICs = pd.DataFrame(data=X_transformed, index=df.index, columns=column_names)
    return make_dataset(df_ICs, omic="IC", database="ENGINEERED")


def get_RPCs(df, n_components: int = 2, label: str = None, random_state=42):
    """returns the PCs columns corresponding to the ICA components of the dataframe"""
    if label is None:
        label = (datetime.now()).strftime("%Y%m%d%H%M%S")
    transformer = GaussianRandomProjection(
        n_components=n_components, random_state=random_state
    )
    X_transformed = transformer.fit_transform(df)
    column_names = []
    for n in range(1, n_components + 1):
        column_names.append("RPC" + str(n) + label)
    df_RPCs = pd.DataFrame(data=X_transformed, index=df.index, columns=column_names)
    return make_dataset(df_RPCs, omic="RPC", database="ENGINEERED")


def get_TSNEs(df, n_components: int = None, label: str = None, random_state=42):
    """returns the t-SNE components of the dataframe"""
    if label is None:
        label = (datetime.now()).strftime("%Y%m%d%H%M%S")
    if n_components is None:
        n_components = 2
    df = df.dropna()
    tsne = TSNE(n_components=n_components, verbose=1, random_state=random_state)
    tsne_components = tsne.fit_transform(df)
    column_names = []
    for n in range(1, n_components + 1):
        column_names.append("TSNE" + str(n) + label)
    df_TSNEs = pd.DataFrame(data=tsne_components, index=df.index, columns=column_names)
    return make_dataset(df_TSNEs, omic="TSNE", database="ENGINEERED")
