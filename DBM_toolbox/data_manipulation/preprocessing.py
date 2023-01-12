# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import logging
from DBM_toolbox.data_manipulation import dataset_class
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from sklearn.impute import KNNImputer

pd.options.mode.chained_assignment = None


def reformat_drugs(dataset):
    """reshapes a CCLE pandas dataframe from 'one line per datapoint' to a more convenient
    'one line per sample' format, meaning the response of a given cell line to different drugs
    will be placed on the same line in different columns."""

    dataframe = dataset.dataframe
    database = dataset.database
    omic = dataset.omic
    if all(x == database[0] for x in database) and all(
        x.split("_")[0] == "DRUGS" for x in omic
    ):
        if database[0] == "CCLE":
            drugNames = dataframe["Compound"].unique()
            dataframe["Compound"].value_counts()
            # concatenate the drug info with one line per cell line
            merged = pd.DataFrame()
            for thisDrug in drugNames:
                dataframe_spec = dataframe.loc[dataframe["Compound"] == thisDrug]
                dataframe_spec_clean = dataframe_spec.drop(
                    columns=[
                        "Primary Cell Line Name",
                        "Compound",
                        "Target",
                        "Activity SD",
                        "Num Data",
                        "FitType",
                    ]
                )
                dataframe_spec_clean.columns = [
                    "CCLE Cell Line Name",
                    thisDrug + "_dr_doses",
                    thisDrug + "_dr_responses",
                    thisDrug + "_EC50",
                    thisDrug + "_IC50",
                    thisDrug + "_Amax",
                    thisDrug + "_ActArea",
                ]

                if merged.empty:
                    merged = dataframe_spec_clean.copy()
                else:
                    merged = pd.merge(
                        merged,
                        dataframe_spec_clean,
                        how="left",
                        on="CCLE Cell Line Name",
                        sort=False,
                        suffixes=("_x", "_y"),
                        copy=True,
                    )
            merged_dataframe = merged.set_index("CCLE Cell Line Name")
            n_rows, n_cols = merged_dataframe.shape
            omic = pd.Series(
                data=["DRUGS" for x in range(n_cols)], index=merged_dataframe.columns
            )
            database = pd.Series(
                data=["CCLE" for x in range(n_cols)], index=merged_dataframe.columns
            )

        elif database[0] == "GDSC":
            pass
    return dataset_class.Dataset(merged_dataframe, omic=omic, database=database)


def preprocess_data(dataset, flag: str = None):
    omic = dataset.omic
    database = dataset.database
    if all(x == database[0] for x in database) and all(x == omic[0] for x in omic):
        if database[0] == "CCLE":
            if omic[0] == "RPPA":
                dataset = preprocess_ccle_rppa(dataset, flag=flag)
            elif omic[0] == "RNA":
                dataset = preprocess_ccle_rna(dataset, flag=flag)
            elif omic[0] == "RNA-FILTERED":
                dataset = preprocess_ccle_rna_filtered(dataset, flag=flag)
            elif omic[0] == "MIRNA":
                dataset = preprocess_ccle_mirna(dataset, flag=flag)
            elif omic[0] == "META":
                dataset = preprocess_ccle_meta(dataset, flag=flag)
            elif omic[0] == "DNA":
                dataset = preprocess_ccle_dna(dataset, flag=flag)
            elif omic[0] == "CHROMATIN":
                dataset = preprocess_ccle_chromatin(dataset, flag=flag)
            else:
                pass
        elif database[0] == "GDSC":
            if omic[0] == "RNA":
                dataset = preprocess_gdsc_rna(dataset, flag=flag)
            if omic[0] == "MIRNA":
                dataset = preprocess_gdsc_mirna(dataset, flag=flag)
            if omic[0] == "DNA":
                dataset = preprocess_gdsc_dna(dataset, flag=flag)
        elif database[0] == "OWN":
            if omic[0] == "PATHWAYS":
                dataset = preprocess_features_pathway(dataset, flag=flag)
            if omic[0] == "EIGENVECTOR":
                dataset = preprocess_features_eigenvector(dataset, flag=flag)
            if omic[0] == "BETWEENNESS":
                dataset = preprocess_features_betweenness(dataset, flag=flag)
            if omic[0] == "CLOSENESS":
                dataset = preprocess_features_closeness(dataset, flag=flag)
            if omic[0] == "PAGERANK":
                dataset = preprocess_features_pagerank(dataset, flag=flag)
            if omic[0] == "AVNEIGHBOUR":
                dataset = preprocess_features_avneighbour(dataset, flag=flag)

    return dataset


def preprocess_ccle_rppa(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df = df.set_index("Unnamed: 0")
        df = rescale_data(df)
        df = np.log2(df + 1)

    return dataset_class.Dataset(df, omic="RPPA", database="CCLE")


def preprocess_ccle_rna(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df["GeneTrans"] = df["Description"] + "_" + df["Name"]
        df = df.set_index(["GeneTrans"])
        df = df.drop(["Description", "Name"], axis=1)
        df = df.transpose()
        df = np.log2(df + 1)

    return dataset_class.Dataset(df, omic="RNA", database="CCLE")


def preprocess_ccle_rna_filtered(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df = df.set_index("Unnamed: 0")
        return dataset_class.Dataset(df, omic="RNA-FILTERED", database="CCLE")


def preprocess_ccle_mirna(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df["GeneTrans"] = df["Description"] + "_" + df["Name"]
        df = df.set_index(["GeneTrans"])
        df = df.drop(["Description", "Name"], axis=1)
        df = df.transpose()
        df = np.log2(df + 1)
        df = rescale_data(df)

    return dataset_class.Dataset(df, omic="MIRNA", database="CCLE")


def preprocess_ccle_meta(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df = df.drop("DepMap_ID", axis=1).set_index(["CCLE_ID"])
    df = rescale_data(df)
    return dataset_class.Dataset(df, omic="META", database="CCLE")


def preprocess_ccle_dna(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df = df.drop("Description", axis=1)
        df = df.set_index("Name")
        df = df.transpose()
        df = rescale_data(df)
    return dataset_class.Dataset(df, omic="DNA", database="CCLE")


def preprocess_ccle_chromatin(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df = df
    df = np.log2(df + 1)
    df = rescale_data(df)
    return dataset_class.Dataset(df, omic="CHROMATIN", database="CCLE")


def preprocess_gdsc_rna(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df = df
        # df['GeneTrans'] = df['Description'] + '_' + df['Name']
        # df = df.set_index(['GeneTrans'])
        # df = df.drop(['Description', 'Name'], axis=1)
        #  df = df.transpose()
    df = np.log2(df + 1)
    df = rescale_data(df)
    return dataset_class.Dataset(df, omic="RNA", database="GDSC")


def preprocess_gdsc_mirna(dataset, flag: str = None):
    # TODO: preprocessing steps here
    pass


def preprocess_gdsc_dna(dataset, flag: str = None):
    # TODO: preprocessing steps here
    pass


def preprocess_features_pathway(dataset, flag: str = None):
    df = dataset.dataframe
    if flag is None:
        df = df.set_index(["Cell_line"])
    return dataset_class.Dataset(df, omic="PATHWAYS", database="OWN")


def preprocess_features_eigenvector(dataset, flag: str = None):
    df = dataset.dataframe
    df = df.drop("Unnamed: 0", axis=1).set_index(["Gene"]).transpose()
    #    df.index = [idx[6:-11] for idx in df.index]
    df.index = [idx.rsplit("_", 1)[0].split("_", 1)[1] for idx in df.index]
    df = df.add_suffix("_topo_eig")
    df = impute_missing_data(df, method="zeros")
    df = impute_missing_data(df, method="zeros", threshold=0.9)

    # additional steps if necessary
    return dataset_class.Dataset(df, omic="EIGENVECTOR", database="OWN")


def preprocess_features_betweenness(dataset, flag: str = None):
    # @Apurva
    df = dataset.dataframe
    df = df.drop("Unnamed: 0", axis=1).set_index(["Gene"]).transpose()
    df.index = [idx[12:-11] for idx in df.index]
    df.index = [idx.rsplit("_", 1)[0].split("_", 1)[1] for idx in df.index]
    df = df.add_suffix("_topo_bet")
    df = impute_missing_data(df, method="zeros", threshold=0.9)
    # additional steps if necessary
    return dataset_class.Dataset(df, omic="BETWEENNESS", database="OWN")


def preprocess_features_closeness(dataset, flag: str = None):
    # @Apurva
    df = dataset.dataframe
    df = df.drop("Unnamed: 0", axis=1).set_index(["Gene"]).transpose()
    #     df.index = [idx[10:-11] for idx in df.index]
    df.index = [idx.rsplit("_", 1)[0].split("_", 1)[1] for idx in df.index]
    df = df.add_suffix("_topo_clo")
    #     df = impute_missing_data(df, method='zeros')
    # additional steps if necessary
    return dataset_class.Dataset(df, omic="CLOSENESS", database="OWN")


def preprocess_features_pagerank(dataset, flag: str = None):
    # @Apurva
    df = dataset.dataframe
    df = df.drop("Unnamed: 0", axis=1).set_index(["Gene"]).transpose()
    #     df.index = [idx[10:-11] for idx in df.index]
    df.index = [idx.rsplit("_", 1)[0].split("_", 1)[1] for idx in df.index]
    df = df.add_suffix("_topo_pgrk")
    #     df = impute_missing_data(df, method='zeros')
    # additional steps if necessary
    return dataset_class.Dataset(df, omic="PAGERANK", database="OWN")


def preprocess_features_avneighbour(dataset, flag: str = None):
    # @Apurva
    df = dataset.dataframe
    df = df.drop("Unnamed: 0", axis=1).set_index(["Gene"]).transpose()
    #     df.index = [idx[10:-11] for idx in df.index]
    df.index = [idx.split("_", 2)[2].rsplit("_", 1)[0] for idx in df.index]
    df = df.add_suffix("_topo_avngb")
    #     df = impute_missing_data(df, method='zeros')
    # additional steps if necessary
    return dataset_class.Dataset(df, omic="AVNEIGHBOUR", database="OWN")


# add more functions here for each dataset


def rescale_data(dataframe):
    """Normalization by mapping to the [0 1] interval (each feature independently)
    this is the same as maxScaler? should we leave it?"""
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())


def impute_missing_data(dataframe, method: str = "average", threshold: float = None):
    """imputes computed values for missing data according to the specified method
    this will apply the same imputation method for all columns"""
    assert isinstance(dataframe, pd.DataFrame), "dataframe must be a pandas DataFrame"
    valid_methods = ["average", "null", "median", "neighbor", "zeros"]
    assert method in valid_methods, f"method must be one of {valid_methods}"
    assert isinstance(threshold, float) and 0 <= threshold <= 1, "threshold must be a float between 0 and 1"

    if threshold is not None:
        df_copy = dataframe.copy()
        df_sum_missing = df_copy.isna().sum(axis=1)
        df_shape = df_copy.shape[1]
        frac_data = 1 - (df_sum_missing / df_shape)
        df_copy["frac"] = frac_data
        new_dataframe = df_copy[df_copy["frac"] > threshold].drop(columns=["frac"])
        df_unselected = df_copy[df_copy["frac"] <= threshold].drop(columns=["frac"])
    else:
        new_dataframe = dataframe
        df_unselected = None
    if method == 'average':
        new_dataframe = new_dataframe.fillna(new_dataframe.mean())
    elif method == "null" or method == 'zeros':
        new_dataframe = new_dataframe.fillna(0)
    elif method == 'median':
        new_dataframe = new_dataframe.fillna(new_dataframe.median())
    elif method == 'neighbor':
        imputer = KNNImputer()
        imputer.fit(new_dataframe)
        new_dataframe = imputer.transform(new_dataframe)
    if threshold is not None:
        new_dataframe = pd.concat([new_dataframe, df_unselected])

    was_na = dataframe.isna().sum(axis=1).sum()
    is_na = new_dataframe.isna().sum(axis=1).sum()
    diff_na = was_na - is_na
    logging.info(f"imputed {diff_na} samples")

    return new_dataframe


def remove_constant_data(dataframe):
    """removes the columns that are strictly constant"""
    new_dataframe = dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]
    return new_dataframe


def get_tumor_type(dataframe):
    tumors_list = [
        "PROSTATE",
        "STOMACH",
        "URINARY",
        "NERVOUS",
        "OVARY",
        "HAEMATOPOIETIC",
        "KIDNEY",
        "THYROID",
        "SKIN",
        "SOFT_TISSUE",
        "SALIVARY",
        "LUNG",
        "BONE",
        "PLEURA",
        "ENDOMETRIUM",
        "BREAST",
        "PANCREAS",
        "AERODIGESTIVE",
        "LARGE_INTESTINE",
        "GANGLIA",
        "OESOPHAGUS",
        "FIBROBLAST",
        "CERVIX",
        "LIVER",
        "BILIARY",
        "SMALL_INTESTINE",
    ]

    dataframe_tumors = pd.DataFrame(index=dataframe.index, columns=tumors_list)
    for this_tumor_type in tumors_list:
        for this_sample in dataframe.index:
            if this_tumor_type in this_sample:
                dataframe_tumors.loc[this_sample, this_tumor_type] = 1.0
            else:
                dataframe_tumors.loc[this_sample, this_tumor_type] = 0.0

    for col in dataframe_tumors.columns:
        dataframe_tumors[col] = pd.to_numeric(dataframe_tumors[col])

    return dataframe_tumors


def select_drug_metric(dataset, metric: str):
    omic = dataset.omic
    database = dataset.database
    dataframe = dataset.dataframe
    is_selected = dataframe.columns.str.contains(metric, regex=False)
    dataframe = dataframe.loc[:, is_selected]
    omic = omic.loc[is_selected]
    database = database.loc[is_selected]
    sparse_dataset = dataset_class.Dataset(dataframe, omic=omic, database=database)

    return sparse_dataset


def reduce_mem_usage(df, check=False):
    """
    Reduces memory usage for large pandas dataframes by changing datatypes per column into the ones
    that need the least number of bytes (int8 if possible, otherwise int16 etc...)
    :param df: pandas DataFrame
    :param check: whether to check the consistency of the data before and after optimization
    :return: the dataframe with reduced memory usage
    """

    # check if input is a pandas dataframe
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"

    # make a copy of the dataframe
    df_original = df.copy()

    # calculate the original memory usage
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info("Memory usage is {:.2f} MB".format(start_mem))

    # iterate over columns
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue

        # check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # get the minimum and maximum value of the column
            c_min = df[col].min()
            c_max = df[col].max()

            # find the numpy dtype with the smallest memory footprint that can hold the maximum value of the column
            dtype = np.min_scalar_type(c_max)

            # change the column dtype
            df[col] = df[col].astype(dtype)

        else:
            df[col] = df[col].astype("category")

    # calculate the new memory usage
    end_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info("Memory usage after optimization is {:.2f} MB".format(end_mem))

    if check:
        df_diffs = pd.DataFrame()
        logging.info("checking consistency...")
        for col in df:
            col_type = df[col].dtype
            if col_type != object:
                df_diffs[col] = df_original[col] - df[col]

        # Mean, max and min for all columns should be 0
        mean_test = df_diffs.describe().loc["mean"].mean()
        max_test = df_diffs.describe().loc["max"].max()
        min_test = df_diffs.describe().loc["min"].min()

        logging.info("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
        logging.info("Min, Max and Mean of pre/post differences: {:.2f}, {:.2f}, {:.2f}".format(
                min_test, max_test, mean_test))

    return df


def extract_ActAreas(dataset):
    dataframe = dataset.dataframe
    cols = dataframe.columns.str.contains("ActArea")
    dataframe = dataframe.loc[:, cols]

    return dataframe


def extract_IC50s(dataset):
    dataframe = dataset.dataframe
    cols = dataframe.columns.str.contains("IC50")
    new_dataframe = dataframe.loc[:, cols]

    return new_dataframe


def extract_dr(dataset):
    dataframe = dataset.dataframe
    cols = dataframe.columns.str.contains("_dr_")
    new_dataframe = dataframe.loc[:, cols]

    # split the strings of values into numpy arrays
    new_dataframe = new_dataframe.apply(lambda x: x.str.split(',').apply(lambda x: list(map(float, x)) if x.dtype == object else x))

    return new_dataframe
