
from functions import unpickle_objects
from config import Config
import pandas as pd
from DBM_toolbox.modeling import validation
from DBM_toolbox.data_manipulation import dataset_class


config = Config("testmin/second/config.yaml")

loo_preds = unpickle_objects("f_testsmall_preds_2022-07-21-12-33-52-564134.pkl")

loo_data = unpickle_objects("f_testsmall_data_2022-07-18-09-10-35-582909.pkl")

drugs_list = ['Lapatinib', 'AZD6244']

global_models = dict()

for drug in drugs_list:
    df = loo_preds.dataframe
    cols = df.columns.str.contains(drug)
    df = df.loc[:, cols].astype(float)
    omics = pd.Series(['prediction'] * (df.shape[1] - 1) + ['DRUGS'], index=df.columns)
    loo_data = dataset_class.Dataset(dataframe=df, omic=omics, database='OWN')
    global_models[drug] = config.get_models(dataset=loo_data, method="standard")

    filename = drug + 'loo_preds.csv'
    df.to_csv(filename)

    models = global_models[drug][df.columns[-1]]['complete']
    models_df = pd.DataFrame(models)
    filename = drug + 'loo-models.csv'
    models_df.to_csv(filename)

    preds = validation.loo(loo_data, algos=['RFC', 'XGB'], metric='roc_auc', targets_list=[df.columns[-1]])
    filename = drug + 'loo_final_pred.csv'
    preds.dataframe.to_csv(filename)

