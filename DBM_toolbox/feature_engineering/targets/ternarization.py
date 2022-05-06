# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:58:05 2020

@author: sebde
"""
# import numpy as np
import pandas as pd

from sklearn.preprocessing import binarize

# from scipy.signal import find_peaks
# from scipy.stats.kde import gaussian_kde
# from astropy import modeling


def get_drug_response(df, thresholdR=0, thresholdS=0, axis="columns"):
    """replaces the data with indication of sensitivity based on quantiles:
        -1 = Resistant
        0 = Intermediate
        1 = Sensitive
        values in df indicate response to drug (like Act Area)
        """
    if isinstance(thresholdS, int):
        dfSens = binarize(
            df, threshold=thresholdS
        )  # Resistants will get 0, while Intermediates and Sensitives get 1
    else:
        dfSens = binarize(df.sub(thresholdS, axis=axis))
    if isinstance(thresholdR, int):
        dfNotR = binarize(
            df, threshold=thresholdR
        )  # Sensitives get 1 more, others get 0
    else:
        dfNotR = binarize(df.sub(thresholdR, axis=axis))
    dfAll = dfSens + dfNotR - 1
    dfAll = pd.DataFrame(data=dfAll, index=df.index, columns=df.columns)
    return dfAll


def ternarize_targets_density(df,):
    """replaces the data with indication of sensitivity based on density:
        -1 = Resistant
        0 = Intermediate
        1 = Sensitive
        values in df indicate response to drug (like Act Area)
        """
    # TODO: integrate Maria's ternarization
    pass
