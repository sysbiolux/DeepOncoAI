import numpy as np
import pandas as pd
import logging
from DBM_toolbox.data_manipulation.filter_class import KeepFeaturesFilter
from DBM_toolbox.data_manipulation import dataset_class
import xgboost as xgb

from sklearn.metrics import mean_squared_error


class Rule:
    def create_filter(self, dataset):
        pass


# TODO: these rules could also be database-specific
class HighestVarianceRule(Rule):
    def __init__(self, fraction, omic, database):
        if fraction < 0 or fraction > 1:
            raise ValueError("HighestVarianceRule fraction should be in [0, 1]")
        self.fraction = fraction
        self.omic = omic
        self.database = database

    def create_filter(self, dataset):
        dataframe = dataset.to_pandas(omic=self.omic)
        variances = dataframe.var().sort_values(ascending=False)
        number_of_features_to_keep = int(round(len(variances) * self.fraction))
        features_to_keep = variances.iloc[:number_of_features_to_keep].index
        print(f"Keeping {len(features_to_keep)} features out of {dataframe.shape[1]}")
        return KeepFeaturesFilter(
            ftype="HighestVariance",
            features=features_to_keep,
            omic=self.omic,
            database=self.database,
        )


class ColumnDensityRule(Rule):
    def __init__(self, completeness_threshold: float, omic: str, database: str):
        if completeness_threshold < 0 or completeness_threshold > 1:
            raise ValueError(
                "ColumnDensityRule completeness_threshold should be in [0, 1]"
            )
        self.completeness_threshold = completeness_threshold
        self.omic = omic
        self.database = database

    def create_filter(self, dataset):
        dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
        # print(f"oooooooooooooooooooo {dataframe.shape[0]}")
        dataframe = dataframe.dropna(how="all")
        # print(f"oooooooooooooooooooo {dataframe.shape[0]}")
        completeness = 1 - dataframe.isna().mean(axis=0)
        features_to_keep = completeness[
            completeness >= self.completeness_threshold
        ].index
        print(f"Keeping {len(features_to_keep)} features out of {dataframe.shape[1]}")
        return KeepFeaturesFilter(
            ftype="ColumnDensity",
            features=features_to_keep,
            omic=self.omic,
            database=self.database,
        )


class CrossCorrelationRule(Rule):
    def __init__(self, correlation_threshold, omic, database):
        if correlation_threshold < 0 or correlation_threshold > 1:
            raise ValueError(
                "CrossCorrelationRule correlation_threshold should be in [0, 1]"
            )
        self.correlation_threshold = correlation_threshold
        self.omic = omic
        self.database = database

    def create_filter(self, dataset):
        dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
        this_dataset = dataset_class.Dataset(
            dataframe=dataframe, omic=self.omic, database=self.database
        )
        dataframe - this_dataset.remove_constants().normalize().impute().to_pandas()
        counter = 0
        for this_feature in dataframe.columns:
            counter += 1
            print(f"{counter}: {this_feature}")
            try:
                dataframe_rest = dataframe.drop(this_feature, axis=1)
                to_compare = dataframe[this_feature]
                corr_single = abs(dataframe_rest.corrwith(to_compare))
                if any(corr_single > self.correlation_threshold):
                    high_corr = corr_single[
                        corr_single > self.correlation_threshold
                    ].index.union([this_feature])
                    logging.info(f"Found high-correlation features: {high_corr}")
                    A = dataframe.drop(high_corr, axis=1)
                    B = dataframe[high_corr]
                    Az = A - A.mean()
                    Bz = B - B.mean()
                    # efficient way to compute cross-correlations?
                    x = (
                        Az.T.dot(Bz)
                        .div(len(A))
                        .div(Bz.std(ddof=0))
                        .div(Az.std(ddof=0), axis=0)
                    )
                    corr_mult = abs(x)
                    scores = (corr_mult * corr_mult).mean()
                    most_correl = scores[scores != max(scores)].index
                    dataframe = dataframe.drop(most_correl, axis=1)
            except:
                print("Feature not present")
        return KeepFeaturesFilter(
            ftype="CrossCorrelation",
            features=dataframe.columns,
            omic=self.omic,
            database=self.database,
        )


class FeatureImportanceRule(Rule):
    def __init__(self, ftype, fraction, omic, database):
        # TODO: Add check on fraction allowed values
        self.fraction = fraction
        self.omic = omic
        self.database = database

    def create_filter(self, dataset, target_dataframe):
        dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
        if len(target_dataframe.shape) == 1:
            index1 = target_dataframe.index[
                target_dataframe.apply(np.isnan)
            ]  ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost
            index2 = dataframe.index[dataframe.apply(np.isnan).any(axis=1)]  ## SOLVED?
            indices_to_drop = index1.union(index2)

            dataframe = dataframe.drop(indices_to_drop)
            target_dataframe = target_dataframe.drop(indices_to_drop)

        importances = pd.DataFrame()
        model = xgb.XGBClassifier(
            max_depth=4, n_estimators=100, colsample_bytree=0.5
        )  ### deeper?

        model.fit(dataframe, target_dataframe)  # use ravel() here?
        scores = pd.Series(
            data=model.feature_importances_,
            name=target_dataframe.name,
            index=dataframe.columns,
        )
        importances = pd.concat([importances, scores], axis=1)
        importances = importances.mean(axis=1).sort_values(ascending=False)

        number_of_features_to_keep = int(round(len(importances) * self.fraction))
        features_to_keep = importances.iloc[:number_of_features_to_keep].index
        return KeepFeaturesFilter(
            ftype="FeatureImportance",
            features=features_to_keep,
            omic=self.omic,
            database=self.database,
        )


class FeaturePredictivityRule(Rule):
    def __init__(self, fraction, omic, database):
        # TODO: Add check on fraction allowed values
        self.fraction = fraction
        self.omic = omic
        self.database = database

    def create_filter(self, dataset, target_dataframe):
        dataframe = dataset.to_pandas(omic=self.omic, database=self.database)
        if (
            len(target_dataframe.shape) == 1
        ):  ### utility to do this (takes a long time) on several targets?
            target_dataframe = target_dataframe.to_frame().dropna()
        #         index = target_dataframe.index[target_df.apply(np.isnan)]   ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost so we need to drop them
        #         to_drop = index.values.tolist()
        #         dataframe = dataframe.drop(to_drop)
        #         target_dataframe = target_dataframe.drop(to_drop)
        predictivities = pd.DataFrame()
        for this_target in target_dataframe.columns:
            model = xgb.XGBRegressor(
                max_depth=4, n_estimators=100, colsample_bytree=0.5
            )  ### deeper?
            model.fit(dataframe, target_dataframe[this_target])
            predicted = model.predict(dataframe)
            base_error = mean_squared_error(target_dataframe[this_target], predicted)
            base_df = dataframe.copy()
            this_target_predictivity = []
            for this_feature in dataframe.columns:
                shuffled_df = base_df.copy()
                shuffled_df[this_feature] = np.random.permutation(
                    shuffled_df[this_feature].values
                )
                model = xgb.XGBRegressor(
                    max_depth=4, n_estimators=100, colsample_bytree=0.5
                )  ### deeper?
                model.fit(dataframe, target_dataframe[this_target])
                shuffled_predicted = model.predict(shuffled_df)
                shuffled_error = mean_squared_error(
                    target_dataframe[this_target], shuffled_predicted
                )
                this_target_predictivity.append(base_error - shuffled_error)
            target_predictivity = pd.Series(
                data=this_target_predictivity, name=this_target, index=dataframe.columns
            )
            predictivities = pd.concat([predictivities, target_predictivity], axis=1)
        predictivities = predictivities.mean(axis=1).sort_values(ascending=True)

        number_of_features_to_keep = int(round(len(predictivities) * self.fraction))
        features_to_keep = predictivities.iloc[:number_of_features_to_keep].index
        return KeepFeaturesFilter(
            ftype="FeaturePredictivity",
            features=features_to_keep,
            omic=self.omic,
            database=self.database,
        )
