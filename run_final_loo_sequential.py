import pandas as pd

from config import Config
from functions import unpickle_objects
from DBM_toolbox.modeling import validation
config = Config("testall/config.yaml")

for n in range(23):
    idx = str(n+1)
    if len(idx) == 1:
        idx = '0' + idx

    final_data = unpickle_objects('f_testall_data_' + idx + '.pkl')
    algos = config.raw_dict["modeling"]["general"]["algorithms"]
    metric = config.raw_dict["modeling"]["general"]["metric"]

    target = final_data.dataframe.columns[-1]

    loo_preds = validation.loo(final_data, algos=algos, metric=metric, targets_list=[target])
    config.save(to_save=loo_preds, name="f_testall_01_preds_" + idx)

##################################
##
##################################

### result analysis

import os
from functions import unpickle_objects
import seaborn as sns
import matplotlib.pyplot as plt

results_df = pd.DataFrame(columns=['Logistic', 'SVC', 'SVM', 'Ridge', 'Ada', 'EN', 'ET', 'XGB', 'RFC', 'KNN'])
files_list = []
cwd = os.getcwd()
for file in os.listdir(cwd):
    if file.startswith("f_testall_01_preds_"):
        files_list.append(file)
        results = unpickle_objects(file)
        target_name = results.dataframe.columns[-1]
        trained_models = config.get_models(dataset=results, method="standard")
        for algo in trained_models[target_name]['complete']:
            perf = trained_models[target_name]['complete'][algo]['result']
            results_df.loc[target_name, algo] = perf


f, ax = plt.subplots()
sns.heatmap(results_df.astype(float), annot=True, ax=ax)
plt.savefig('global_results_2Dloo.svg')

######################################
## same but with 10x nested CV
######################################



import pandas as pd
import numpy as np
rng = np.random.default_rng()

from config import Config
from functions import unpickle_objects
from DBM_toolbox.modeling import validation
config = Config("testall/config.yaml")


outer_folds = 10
inner_folds = 10


for n in range(23):
    idx = str(n+1)
    if len(idx) == 1:
        idx = '0' + idx

    final_data = unpickle_objects('f_testall_data_' + idx + '.pkl')
    algos = config.raw_dict["modeling"]["general"]["algorithms"]
    metric = config.raw_dict["modeling"]["general"]["metric"]
    target_name = final_data.dataframe.columns[-1]
    final_data = final_data.to_binary(target=target_name)

    n_samples = len(final_data.dataframe.index)
    split_size = np.floor(n_samples/outer_folds)
    outer_mixed = rng.choice(final_data.dataframe.index, size=n_samples, replace=False)

    outer_train_idx = dict()
    outer_test_idx = dict()
    inner_train_idx = dict()
    inner_test_idx = dict()

    for outer_loop in range(outer_folds):

        if outer_loop == outer_folds - 1:
            outer_test_idx[outer_loop] = outer_mixed[int(outer_loop * split_size): -1]
        else:
            outer_test_idx[outer_loop] = outer_mixed[int(outer_loop * split_size): int((outer_loop + 1) * split_size)]
        outer_train_idx[outer_loop] = [x for x in outer_mixed if x not in outer_test_idx[outer_loop]]

        rest_dataset, valid_dataset = final_data.split(train_index=outer_train_idx[outer_loop], test_index=outer_test_idx[outer_loop])

        n_inner_samples = len(rest_dataset.dataframe.index)
        split_inner_size = np.floor(n_inner_samples / inner_folds)
        inner_mixed = rng.choice(rest_dataset.dataframe.index, size=n_inner_samples, replace=False)

        inner_train_idx[outer_loop] = dict()
        inner_test_idx[outer_loop] = dict()

        for inner_loop in range(inner_folds):
            print(f"n: {n}, out: {outer_loop}, in: {inner_loop}")
            if inner_loop == inner_folds - 1:
                inner_test_idx[outer_loop][inner_loop] = inner_mixed[int(inner_loop * split_inner_size): -1]
            else:
                inner_test_idx[outer_loop][inner_loop] = inner_mixed[int(inner_loop * split_inner_size): int((inner_loop + 1) * split_inner_size)]
            inner_train_idx[outer_loop][inner_loop] = [x for x in inner_mixed if x not in inner_test_idx[outer_loop][inner_loop]]

            train_dataset, test_dataset = rest_dataset.split(train_index=inner_train_idx[outer_loop][inner_loop], test_index=inner_test_idx[outer_loop][inner_loop])

