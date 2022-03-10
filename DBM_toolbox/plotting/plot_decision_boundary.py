# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:15:13 2020

@author: sebde
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def plot_decision_boundary(
    y_test, X_test, y_train, X_train, thisNames, models, item=0, ax=None, pca=None
):
    if ax == None:
        fig, ax = plt.subplots()
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    clf = models[thisNames[item]]
    clf.fit(X, y)
    pca = PCA(n_components=2)
    pca = pca.fit(X)
    PCs = pca.transform(X)
    Xfake = pd.concat([X.copy(), X.copy(), X.copy()], ignore_index=True)
    for col in Xfake.columns:
        val = Xfake[col].values
        np.random.shuffle(val)
        Xfake[col] = val

    Z = clf.predict(Xfake)
    ZPCs = pca.transform(Xfake)
    missed = clf.predict(X_test) != y_test

    ax.scatter(
        ZPCs[:, 0], ZPCs[:, 1], c=Z, s=5, cmap="coolwarm", alpha=0.5, linewidths=0
    )
    # 	ax.scatter(PCs[:,0], PCs[:,1], c=y, s=20, cmap='bwr', edgecolor='k')
    ax.set_title("Decision boundary")
    plt.show()
    return ax
