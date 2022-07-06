import pandas as pd
import logging

from DBM_toolbox.modeling import optimized_models
from DBM_toolbox.data_manipulation import data_utils
# import numpy as np


def loo(dataset, algos, metric, targets_dict):
    dataframe = dataset.dataframe
    omic = dataset.omic
    omics_unique = list(set(omic))
    omics_unique.remove('DRUGS')
    # database = dataset.database

    targets_list = []
    for item in targets_dict:
        this_name = item["target_drug_name"] + "_" + item["responses"]
        targets_list.append(this_name)
    targets_list = list(set(targets_list))
    colnames = []
    for this_omic in omics_unique:
        for this_target_name in targets_list:
            for algo in algos:
                colnames.append(this_omic + "_" + this_target_name + "_" + algo)
    preds = pd.DataFrame(index=dataframe.index, columns=colnames)
    for this_omic in omics_unique:
        print(this_omic)
        for this_target_name in targets_list:
            print(this_target_name)
            this_target = dataset.dataframe.loc[:, this_target_name]
            m = min(this_target)
            n = max(this_target)

            this_target = this_target[this_target.isin([m, n])]
            this_data = dataset.to_pandas(omic=this_omic).loc[this_target.index, :]
            for sample in this_data.index:
                logging.info(f"{this_omic}, {this_target_name}, {sample}")
                print(sample)
                rest_data = this_data.drop(sample)
                rest_target = this_target.drop(sample)
                rest_data, rest_target = data_utils.merge_and_clean(rest_data, rest_target)
                this_result = optimized_models.get_standard_models(
                    data=rest_data, targets=rest_target, algos=algos, metric=metric
                )
                print(this_result)
                for algo in algos:
                    le_model = this_result[algo]['estimator'].fit(rest_data, rest_target)
                    to_predict = this_data.loc[sample, :].to_frame().transpose()
                    try:
                        pred = le_model.predict_proba(to_predict)
                    except:
                        pred = le_model.predict(to_predict)
                    print(pred)
                    print(this_omic)
                    print(this_target_name)
                    print(algo)
                    colname = this_omic + "_" + this_target_name + "_" + algo
                    preds.loc[sample, colname] = pred
    return preds