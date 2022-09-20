from functions import unpickle_objects
import pandas as pd
from DBM_toolbox.data_manipulation import dataset_class
from config import Config
config = Config("testall/config.yaml")

data = unpickle_objects('f_testall_01_data_2022-07-21-21-31-08-689557.pkl')
df = data.dataframe
omic = data.omic
database = data.database

base_idx = omic[omic == 'DRUGS'].index
base_num = list(range(23))

for k, drug_to_keep in enumerate(base_idx):
    to_drop = [x for x in base_idx if x != drug_to_keep]
    new_df = df.drop(to_drop, axis=1, inplace=False)
    new_omic = omic.drop(to_drop, inplace=False)
    new_db = database.drop(to_drop, inplace=False)

    new_dataset = dataset_class.Dataset(dataframe = new_df, omic=new_omic, database=new_db)
    increment = str(k+1)
    if len(increment) == 1:
        increment = '0' + increment
    filename = 'f_testsmall_data_' + increment
    config.save(new_dataset, filename)


