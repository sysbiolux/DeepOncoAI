# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:51:13 2020

@author: sebde
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def explore_shape(df, plot=False):
    """accepts a pandas dataframe and returns its dimensions as well as a
	graph of the presence of data."""

    n_lines = df.shape[0]
    n_columns = df.shape[1]
    fraction_missing = df.isna().mean().mean()
    if plot:
        plt.figure()
        sns.heatmap(df.isnull(), cbar=False)

    print(
        "%.1f samples and %.1f features, of which %.2f percent is missing"
        % (n_lines, n_columns, fraction_missing * 100)
    )
