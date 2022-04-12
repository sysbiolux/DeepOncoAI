import pandas as pd
from DBM_toolbox.data_manipulation.dataset_class import Dataset


class KeepFeaturesFilter:
    def __init__(self, ftype, features, omic, database):
        self.ftype = ftype
        self.features = features
        self.omic = omic
        self.database = database

    def apply(self, dataset):
        # TODO: this takes too much time!
        dataframe = dataset.dataframe
        omic = dataset.omic
        database = dataset.database

        selected = pd.DataFrame(index=omic.index, dtype=bool)
        selected["omic"] = (omic != self.omic).values
        selected["database"] = database != self.database
        selected["retained"] = selected.any(axis=1)
        for this_feature in self.features:
            try:
                selected.loc[this_feature, "retained"] = True
            except:
                pass

        filtered_dataframe = dataframe.loc[:, selected["retained"] == True]
        retained_omic = omic.loc[selected["retained"] == True]
        retained_database = database.loc[selected["retained"] == True]

        return Dataset(
            dataframe=filtered_dataframe, omic=retained_omic, database=retained_database
        )

    def __repr__(self):
        return (
            f"KeepFeaturesFilter with type {self.ftype}, ({self.features}, {self.omic}, {self.database})"
        )


class KeepDenseRowsFilter:
    def __init__(self, ftype, completeness_threshold, omic, database):
        self.ftype = ftype
        self.completeness_threshold = completeness_threshold
        self.omic = omic
        self.database = database

    def apply(self, dataset):
        dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
        completeness = 1 - (dataframe.isna().mean(axis=1))
        samples_to_keep = completeness[
            completeness >= self.completeness_threshold
        ].index
        filtered_dataframe = dataset.dataframe.loc[samples_to_keep, :]
        return Dataset(
            dataframe=filtered_dataframe, omic=dataset.omic, database=dataset.database
        )

    def __repr__(self):
        return (
            f"KeepDenseRowsFilter with type {self.ftype}, ({self.completeness_threshold}, {self.omic}, {self.database})"
        )
