####################
### HOUSEKEEPING ###
####################

import logging
# import numba #does not work?
# @numba.jit
logging.basicConfig(filename='run.log', level=logging.INFO, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
from config import Config
from DBM_toolbox.data_manipulation import dataset_class
config = Config()

###################################
### READING AND PROCESSING DATA ###
###################################

logging.info("Reading data")
data = config.read_data()

# logging.info("Creating visualizations")
# config.visualize_dataset(data, mode='pre')

logging.info("Filtering data")
filtered_data, filters = config.filter_data(data)

print(filtered_data.dataframe.shape)
for omic in list(set(filtered_data.omic)):
    print(f"{omic}: {filtered_data.omic[filtered_data.omic == omic].shape[0]}")

logging.info("Selecting subsets for feature engineering")
selected_subset = config.select_subsets(filtered_data)

logging.info("Engineering features")
if selected_subset is not None:
    engineered_features = config.engineer_features(selected_subset)
    logging.info("Merging engineered features")
    engineered_data = filtered_data.merge_with(engineered_features).normalize()
else:
    engineered_data = filtered_data.normalize()

logging.info("Quantizing targets")
final_data = engineered_data.quantize(target_omic="DRUGS").optimize_formats()

# logging.info("Visualizing distributions")
# config.visualize_dataset(final_data, mode='post')

# logging.info("Getting optimized models")
# optimal_algos_150 = config.get_models(dataset=final_data, method='optimize')
# config.save(to_save=optimal_algos_30, name='optimal_algos_3omics6drugs_30')

logging.info("Getting optimized models")
optimal_algos_30 = config.get_models(dataset=final_data, method='optimize')
config.save(to_save=optimal_algos_30, name='optimal_algos_3omics6drugs_30')

# logging.info("Getting standard models")
# standard_algos = config.get_models(dataset=final_data, method='standard')
# config.save(to_save=standard_algos, name='standard_algos_3omics6drugs')

algos_dict, results_prim = config.get_best_algos(optimal_algos_30)

# config.show_results(results_prim)


#%%

logging.info("Creating best stacks")
best_stacks, results_sec = config.get_best_stacks(models=algos_dict, dataset=final_data)
# algos_dict_over, _ = config.get_best_algos(optimal_algos, mode='over')
# over_stacks, results_over = config.get_best_stacks(models=algos_dict_over, dataset=final_data, tag='_over')

config.save(to_save=best_stacks, name='stack_results')

print('DONE')

#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style(style='whitegrid')
sns.set_context(context='talk')
standard_algos = pd.read_pickle("standard_algos_3omics6drugs_2021-10-01-19-08-27-466682.pkl")
optimal_algos_30 = pd.read_pickle("optimal_algos_3omics6drugs_30_2021-10-03-02-04-23-304011.pkl")
optimal_algos_150 = pd.read_pickle("optimal_algos_3omics6drugs_150_2021-10-02-18-48-30-926074.pkl")

alll = [standard_algos, optimal_algos_30, optimal_algos_150]

idx = 0

t = pd.DataFrame(columns=['method', 'target', 'omic', 'algo', 'perf', 'params'])

for i, this in enumerate(alll):
    targets = this.keys()
    for target in targets:
        print(target)
        omics = this[target].keys()
        for omic in omics:
            algos = this[target][omic].keys()
            for algo in algos:
                r = this[target][omic][algo]['result']
                if type(r) is np.float64:
                    perf = r
                    params = 0
                elif type(r) is dict:
                    perf = r['target']
                    params = r['params']
                else:
                    perf = np.nan
                    params = np.nan
                t.loc[idx, 'method'] = ['standard', 'optimized_30', 'optimized_150'][i]
                t.loc[idx, 'target'] = target.split('_')[0]
                t.loc[idx, 'omic'] = omic
                t.loc[idx, 'algo'] = algo
                t.loc[idx, 'perf'] = perf
                if i == 0:
                    t.loc[idx, 'params'] = params
                else:
                    if type(params) is dict:
                        for param in params.keys():
                            t.loc[idx, param] = params[param]
                idx = idx+1
                

g = sns.catplot(x='algo', y='perf', hue='method', row='target', col='omic', 
            data=t, kind='bar', height=5,
            legend_out=True)
for ax in g.axes.flat:
    ax.set_ylim(0.5, 0.95)
plt.tight_layout()



#%%
targets = t['target'].unique()
for target in targets:
    for omic in omics:
        for algo in algos:
            x = t.loc[(t['target']==target) & (t['omic']==omic) & 
                      (t['algo']==algo),['method','perf']]
            imp1 = x.loc[x['method']=='optimized_30', 'perf'].values - x.loc[x['method']=='standard', 'perf'].values
            imp2 = x.loc[x['method']=='optimized_150', 'perf'].values - x.loc[x['method']=='optimized_30', 'perf'].values
            t.loc[(t['target']==target) & (t['omic']==omic) & 
                  (t['algo']==algo) & (t['method']=='optimized_30'),['improvement_1']] = imp1
            t.loc[(t['target']==target) & (t['omic']==omic) & 
                  (t['algo']==algo) & (t['method']=='optimized_150'),['improvement_2']] = imp2

fig = plt.figure()
sns.distplot(t['improvement_1'], rug=True, color='black')
t['improvement_1'].mean()

fig2 = plt.figure()
sns.distplot(t['improvement_2'], rug=True, color='black')
t['improvement_2'].mean()

t['improvement'] = t[['improvement_1', 'improvement_2']].sum(axis=1, min_count=1)

f, ax = plt.subplots(figsize=(15,10))
sns.swarmplot(x='algo', y='improvement', data=t.loc[t['method']=='optimized_30',:], color='black', ax=ax)
plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(15,10))
sns.swarmplot(x='algo', y='improvement', data=t.loc[t['method']=='optimized_150',:], color='black', ax=ax)
plt.xticks(rotation=90)

f, ax = plt.subplots(figsize=(15,10))
sns.swarmplot(x='omic', y='improvement', data=t.loc[t['method']=='optimized_30',:], color='black', ax=ax)
plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(15,10))
sns.swarmplot(x='omic', y='improvement', data=t.loc[t['method']=='optimized_150',:], color='black', ax=ax)
plt.xticks(rotation=90)

f, ax = plt.subplots(figsize=(15,10))
sns.swarmplot(x='target', y='improvement', data=t.loc[t['method']=='optimized_30',:], color='black', ax=ax)
plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(15,10))
sns.swarmplot(x='target', y='improvement', data=t.loc[t['method']=='optimized_150',:], color='black', ax=ax)
plt.xticks(rotation=90)

#%%
all_omic = final_data.omic.unique().tolist()
all_omic.remove('DRUGS')

for target in algos_dict.keys():
    for omic in algos_dict[target].keys():
        xx = algos_dict[target][omic][0]
        for idx in xx.index:
            model = xx[idx]
            if omic == 'complete':
                X = final_data.extract(omics_list=all_omic).to_pandas()
            else:
                X = final_data.to_pandas(omic=omic)
            
            y = final_data.to_pandas(omic='DRUGS').loc[:, target]
            to_keep = y[y != 0.5].index
            X = X.loc[to_keep, :]
            y = y[to_keep]
            trained = model.fit(X, y)























































