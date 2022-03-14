# -*- coding: utf-8 -*-
"""

"""

from DBM_toolbox.data_manipulation import dataset_class

import pandas as pd


def make_dataset(dataframe, omic=None, database=None):
    dataset = dataset_class.Dataset(dataframe=dataframe, omic=omic, database=database)
    return dataset


def some_fast_topological_analysis(dataframe, parameter="default", label=None):
    # @ Apurva
    ##
    ##
    ##
    engineered_features = dataframe

    return make_dataset(
        dataframe=engineered_features, omic="TOPOLOGY", database="ENGINEERED"
    )
