
import pandas as pd
import numpy as np

from config import Config
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from functions import pickle_objects, unpickle_objects
from DBM_toolbox.data_manipulation import data_utils

config = Config("testall/config.yaml")
algos = config.raw_dict["modeling"]["general"]["algorithms"]
metric = config.raw_dict["modeling"]["general"]["metric"]
rng = np.random.default_rng(42)

final_results = dict()
feature_importances = dict()
base_models = dict()

final_data = unpickle_objects('f_testall_data_01.pkl')

target_name = final_data.dataframe.columns[-1]
print(target_name)
final_data = final_data.to_binary(target=target_name)

n_samples = len(final_data.dataframe.index)
split_size = np.floor(n_samples/10)
outer_mixed = rng.choice(final_data.dataframe.index, size=n_samples, replace=False)

outer_train_idx = dict()
outer_test_idx = dict()
inner_train_idx = dict()
inner_test_idx = dict()
feature_importances[target_name] = dict()
final_results[target_name] = pd.DataFrame(final_data.dataframe.iloc[:, -1].copy())  # truth

trained_models = config.get_models(dataset=final_data, method="standard")

outer_loop = 0

feature_importances[target_name][outer_loop] = dict()
outer_test_idx[outer_loop] = outer_mixed[int(outer_loop * split_size): int((outer_loop + 1) * split_size)]
outer_train_idx[outer_loop] = [x for x in outer_mixed if x not in outer_test_idx[outer_loop]]

rest_dataset, valid_dataset = final_data.split(train_index=outer_train_idx[outer_loop], test_index=outer_test_idx[outer_loop])

n_inner_samples = len(rest_dataset.dataframe.index)
split_inner_size = np.floor(n_inner_samples / 10)
inner_mixed = rng.choice(rest_dataset.dataframe.index, size=n_inner_samples, replace=False)

inner_train_idx[outer_loop] = dict()
inner_test_idx[outer_loop] = dict()
base_models[outer_loop] = dict()

first_level_preds = pd.DataFrame(final_data.dataframe.iloc[:, -1].copy())  # truth

for inner_loop in [0, 1, 2]:

    inner_test_idx[outer_loop][inner_loop] = inner_mixed[int(inner_loop * split_inner_size): int((inner_loop + 1) * split_inner_size)]
    inner_train_idx[outer_loop][inner_loop] = [x for x in inner_mixed if x not in inner_test_idx[outer_loop][inner_loop]]

    train_dataset, test_dataset = rest_dataset.split(train_index=inner_train_idx[outer_loop][inner_loop], test_index=inner_test_idx[outer_loop][inner_loop])

    base_models[outer_loop][inner_loop] = dict()

    for omic in trained_models[target_name].keys():  # for each omic type
        if omic != 'complete':
            print(f'omic: {omic}')
            train_features = train_dataset.to_pandas(omic=omic)
            train_labels = train_dataset.to_pandas(omic='DRUGS').values.ravel()
            test_features = test_dataset.to_pandas(omic=omic)
            test_labels = test_dataset.to_pandas(omic='DRUGS').values.ravel()

            base_models[outer_loop][inner_loop][omic] = dict()

            for algo in trained_models[target_name][omic].keys():  # for each algo
                print(f'algo: {algo}')
                this_model = trained_models[target_name][omic][algo]['estimator']
                this_model.fit(train_features, train_labels)
                try:
                    this_predictions = this_model.predict_proba(test_features)
                    this_predictions = this_predictions[:, 1]
                except:
                    this_predictions = this_model.predict(test_features)
                first_level_preds.loc[
                    inner_test_idx[outer_loop][inner_loop], 'pred_' + omic + '_' + algo] = this_predictions
                try:
                    base_models[outer_loop][inner_loop][omic][algo] = this_model.feature_importances_
                except:
                    try:
                        base_models[outer_loop][inner_loop][omic][algo] = this_model.coef_
                    except:
                        base_models[outer_loop][inner_loop][omic][algo] = 'None'
                base_models[outer_loop][inner_loop][omic]['data'] = train_features















