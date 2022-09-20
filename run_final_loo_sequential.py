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
