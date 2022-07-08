import pandas as pd
import logging

from DBM_toolbox.modeling import optimized_models
from DBM_toolbox.data_manipulation import data_utils, dataset_class
# import numpy as np


def loo(dataset, algos, metric, targets_list):
    dataframe = dataset.dataframe
    omic = dataset.omic
    omics_unique = list(set(omic))
    omics_unique.remove('DRUGS')
    # database = dataset.database

    colnames = []
    for this_omic in omics_unique:
        for this_target_name in targets_list:
            for algo in algos:
                colnames.append(this_omic + "_" + this_target_name + "_" + algo)
    preds_quant = pd.DataFrame(index=dataframe.index, columns=colnames)
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
                    try:
                        pred = data_utils.recurse_to_float(pred)
                    except:
                        print("there was a problem")
                    print(f"{this_target_name} with {this_omic} with {algo}...", end="")
                    print(f"raw pred: {pred}...", end="")
                    print(pred)
                    colname = this_omic + "_" + this_target_name + "_" + algo
                    preds_quant.loc[sample, colname] = pred
    target_data = dataset.to_pandas(omic='DRUGS')
    preds_dataset = dataset_class.Dataset(preds_quant, omic='prediction', database='OWN')
    target_dataset = dataset_class.Dataset(target_data, omic='DRUGS', database='OWN')
    sec_dataset = preds_dataset.merge_two_datasets(target_dataset)
    return sec_dataset


