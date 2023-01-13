import logging

import pandas as pd
import numpy as np
from DBM_toolbox.data_manipulation import preprocessing
from typing import Union

# user class used in this project. Bundles the actual data and traceback of the omic type and database of origin.


class Dataset:
    def __init__(self, dataframe, omic: Union[list, str], database: Union[list, str]):
        self.dataframe = dataframe
        n_rows, n_cols = dataframe.shape
        if isinstance(omic, str):
            self.omic = pd.Series(
                data=[omic for x in range(n_cols)], index=dataframe.columns
            )
        elif len(omic) == n_cols:
            self.omic = omic
        else:
            raise ValueError(
                "Omic should be either a string or a Series with the same lengths as the number of features"
            )
        if isinstance(database, str):
            self.database = pd.Series(
                data=[database for x in range(n_cols)], index=dataframe.columns
            )
        elif len(database) == n_cols:
            self.database = database
        else:
            raise ValueError(
                "Database should be either a string or a Series with the same lengths as the number of features"
            )

    def __str__(self):
        return f"Dataset with omic {self.omic}, from database {self.database}"

    def apply_filters(self, filters: list = None):
        resulting_dataset = self
        if filters:
            for individual_filter in filters:
                logging.info(individual_filter)
                logging.info(f"{individual_filter}")
                size_pre = resulting_dataset.dataframe.shape
                resulting_dataset = individual_filter.apply(resulting_dataset)
                size_post = resulting_dataset.dataframe.shape
                logging.info(f"pre-filter: {size_pre[0]} samples and {size_pre[1]} features")
                logging.info(
                    f"post-filter: {size_post[0]} samples and {size_post[1]} features"
                )
        return resulting_dataset

    def to_pandas(
        self, omic: str = None, database: str = None
    ):  # TODO: possibility to use lists of omics and databases?
        """
        returns the Pandas Dataframe of the dataset for columns matching BOTH the omic and database
        """

        resulting_dataframe = self.dataframe
        resulting_database = self.database
        if omic is not None:
            if omic not in list(self.omic):
                raise ValueError(f"Omics type {omic} not present")
            resulting_dataframe = resulting_dataframe.loc[:, self.omic == omic]
            resulting_database = resulting_database.loc[self.omic == omic]
        if database is not None:
            if database not in list(self.database):
                raise ValueError(f"Database {database} not present")
            resulting_dataframe = resulting_dataframe.loc[
                :, resulting_database == database
            ]
        return resulting_dataframe

    def extract(self, omics_list: list = [], databases_list: list = []):
        """
        returns the parts of the dataset matching EITHER ONE of the elements of the omics_list and databases_list
    
        """
        resulting_dataframe = self.dataframe
        resulting_database = self.database
        resulting_omic = self.omic
        if (omics_list is not None) or (databases_list is not None):
            to_extract = resulting_omic.isin(omics_list) | resulting_database.isin(
                databases_list
            )
            resulting_dataframe = resulting_dataframe.loc[:, to_extract == True]
            resulting_database = resulting_database.loc[to_extract == True]
            resulting_omic = resulting_omic.loc[to_extract == True]
        return Dataset(
            dataframe=resulting_dataframe,
            omic=resulting_omic,
            database=resulting_database,
        )

    def merge_with(self, other_datasets: list):
        if isinstance(other_datasets, list):
            for single_dataset in other_datasets:
                self = self.merge_two_datasets(single_dataset)
        else:
            if isinstance(other_datasets, Dataset):
                self = self.merge_two_datasets(other_datasets)
            else:
                raise ValueError("Merging is only allowed between Datasets")
        return self

    def merge_two_datasets(self, other_dataset):

        dataframe = self.dataframe
        other_dataframe = other_dataset.dataframe

        cols = dataframe.columns

        merged_dataframe = pd.concat([dataframe, other_dataframe], axis=1)
        merged_omic = pd.concat([self.omic, other_dataset.omic])
        merged_database = pd.concat([self.database, other_dataset.database])

        merged_dataframe = merged_dataframe.dropna(how="all", subset=cols)

        merged_dataset = Dataset(
            dataframe=merged_dataframe, omic=merged_omic, database=merged_database
        )

        return merged_dataset

    def impute(self, method: str = "average"):
        return Dataset(
            dataframe=preprocessing.impute_missing_data(self.dataframe, method=method),
            omic=self.omic,
            database=self.database,
        )

    def remove_constants(self):
        n_before = self.dataframe.shape[1]
        dataframe = preprocessing.remove_constant_data(self.dataframe)
        logging.info(
            "Removed " + str(n_before - dataframe.shape[1]) + " constant features"
        )
        return Dataset(
            dataframe=dataframe,
            omic=self.omic[dataframe.columns],
            database=self.database[dataframe.columns],
        )

    def normalize(self):
        return Dataset(
            dataframe=preprocessing.rescale_data(self.dataframe),
            omic=self.omic,
            database=self.database,
        )

    def data_pop_quantize(self, target_omic: str, quantiles_df=None):
        """
        will ternarize the columns based on the population distribution
        quantiles should be a list of two values in [0, 1]
        """
        omic = self.omic
        database = self.database
        dataframe = self.to_pandas()
        if quantiles_df is None:
            quantiles_df = pd.DataFrame(index=["idx"], columns=["low", "high"])
            quantiles_df.loc["idx", :] = [0.333, 0.667]
        elif type(quantiles_df) is tuple:
            quantiles = quantiles_df
            quantiles_df = pd.DataFrame(index=["idx"], columns=["low", "high"])
            quantiles_df.loc["idx", :] = quantiles
        quantized_dataframe = dataframe.copy()
        for target in omic[omic.str.startswith(target_omic)].index:
            quantiles = [
                quantiles_df.loc[target.split("_")[0], "low"],
                quantiles_df.loc[target.split("_")[0], "high"],
            ]
            print(quantiles)
            q = np.quantile(dataframe[target].dropna(), quantiles)
            quantized_dataframe[
                target
            ] = 0.5  # start assigning 'intermediate' to all samples
            quantized_dataframe[target].mask(
                dataframe[target] < q[0], 0, inplace=True
            )  # samples below the first quantile get 0
            quantized_dataframe[target].mask(
                dataframe[target] >= q[1], 1, inplace=True
            )  # samples above get 1

        return Dataset(dataframe=quantized_dataframe, omic=omic, database=database)

    def data_threshold_quantize(self, target_omic: str, ic50s, thresholds):
        omic = self.omic
        database = self.database
        dataframe = self.dataframe
        binarized_ic50s = ic50s.copy()
        t_dataframe = self.to_pandas(omic="DRUGS")
        final_dataframe = t_dataframe.copy()

        for target in ic50s.columns:
            print(target)
            this_threshold = thresholds[target.split("_")[0]]
            if not pd.isna(this_threshold):
                binarized_ic50s[target] = 0.5
                # if -log(IC50) is higher than threshold, then the IC50 is low (sensitive) :
                binarized_ic50s[target].mask(
                    ic50s[target] > this_threshold, 1, inplace=True
                )
                # if -log(IC50) is lower than threshold (resistant) :
                binarized_ic50s[target].mask(
                    ic50s[target] < this_threshold, 0, inplace=True
                )

        for target in t_dataframe.columns:
            if not pd.isna(thresholds[target.split("_")[0]]):
                final_dataframe.loc[:, target] = 0.5
                drug_name = target.split("_")[0]
                IC50_name = drug_name + "_IC50"
                print(f"target: {target}")

                for sample in t_dataframe.index:
                    print("***" * 20)
                    print(f"sample: {sample}, ", end="")
                    if sample in binarized_IC50s.index:
                        print("sample found in IC50s, ", end="")
                        print(
                            f"quantized: {t_dataframe.loc[sample, target]}, IC50: {IC50s.loc[sample, IC50_name]}, binarized_IC50: {binarized_IC50s.loc[sample, IC50_name]}, ",
                            end="",
                        )
                        if (
                            t_dataframe.loc[sample, target]
                            == binarized_IC50s.loc[sample, IC50_name]
                        ):
                            final_dataframe.loc[sample, target] = t_dataframe.loc[
                                sample, target
                            ]
                            print(f"decision: {final_dataframe.loc[sample, target]}")
                    else:
                        print("sample not found in IC50s")

        chosen_omics = [x for x in omic.unique() if x != target_omic]
        left_dataframe = self.extract(omics_list=chosen_omics).dataframe

        dataframe = pd.merge(
            left_dataframe, final_dataframe, left_index=True, right_index=True
        )

        return Dataset(dataframe=dataframe, omic=omic, database=database)

    def optimize_formats(self):
        logging.info("Optimizing formats")
        dataframe = self.dataframe
        dataframe_copy = dataframe.copy()  # EDIT_AHB
        optimal_dataframe = preprocessing.reduce_mem_usage(dataframe_copy)
        return Dataset(
            dataframe=optimal_dataframe, omic=self.omic, database=self.database
        )

    def to_binary(self, target: str):
        omic = self.omic
        database = self.database
        dataframe = self.to_pandas()

        m = min(dataframe[target])
        n = max(dataframe[target])

        dataframe = dataframe[dataframe[target].isin([m, n])]

        return Dataset(dataframe=dataframe, omic=omic, database=database)

    def split(self, train_index: list, test_index: list):
        omic = self.omic
        database = self.database
        dataframe = self.to_pandas()
        train_dataset = Dataset(
            dataframe=dataframe.loc[train_index, :],
            omic=omic,
            database=database,
        )
        test_dataset = Dataset(
            dataframe=dataframe.loc[test_index, :],
            omic=omic,
            database=database,
        )

        return train_dataset, test_dataset

    def change_att(self, old_omic: str = None, new_omic: str = None, old_database: str = None, new_database: str = None):
        if old_omic is not None and new_omic is not None:
            previous = self.omic
            self.omic = pd.Series(new_omic, index=previous.columns)
        if old_database is not None and new_database is not None:
            previous = self.database
            self.database = pd.Series(new_database, index=previous.columns)



