####################
### HOUSEKEEPING ###
####################

import logging

logging.basicConfig(
    filename="run.log",
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
from config import Config

from DBM_toolbox.data_manipulation import dataset_class
from DBM_toolbox.interpretation import gsea

config = Config("testmin/first/config.yaml")

###################################
### READING AND PROCESSING DATA ###
###################################

logging.info("Reading data")
data, ActAreas, IC50s, dose_responses = config.read_data()

# logging.info("Creating visualizations")
# config.visualize_dataset(data, ActAreas, IC50s, dose_responses, mode="pre")

logging.info("Filtering data")
filtered_data, filters = config.filter_data(data)

#####

logging.info("Selecting subsets for feature engineering")
selected_subset = config.select_subsets(filtered_data)

logging.info("Engineering features")
engineered_features = config.engineer_features(filtered_data)

logging.info("Merging engineered features")
engineered_data = filtered_data.merge_with(engineered_features)

logging.info("Quantizing targets")
quantized_data = config.quantize(engineered_data, target_omic="DRUGS", IC50s=IC50s)

final_data = quantized_data.normalize().optimize_formats()

logging.info("Getting optimized models")

trained_models = config.get_models(dataset=final_data, method="standard")
config.save(to_save=final_data, name="f_test2_data")
config.save(to_save=trained_models, name="f_test2_models")

models, algos_dict = config.get_best_algos(trained_models)

# config.show_results(config, algos_dict)

logging.info("Creating best stacks")
results_sec = config.get_stacks(dataset=final_data, models_dict=trained_models)

# algos_dict_over, _ = config.get_best_algos(optimal_algos, mode='over')
# over_stacks, results_over = config.get_best_stacks(models=algos_dict_over, dataset=final_data, tag='_over')

config.save(to_save=results_sec, name="stack_results")

expl_dict = config.retrieve_features(trained_models=trained_models, dataset=final_data)
config.save(to_save=expl_dict, name="expl_dict")



print("DONE")


##################################################################################################################
# compare optimized versus standard

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style(style="whitegrid")
sns.set_context(context="talk")
standard_algos = pd.read_pickle(
    "standard_algos_3omics6drugs_2021-10-01-19-08-27-466682.pkl"
)
optimal_algos_30 = pd.read_pickle(
    "optimal_algos_3omics6drugs_30_2021-10-03-02-04-23-304011.pkl"
)
optimal_algos_150 = pd.read_pickle(
    "optimal_algos_3omics6drugs_150_2021-10-02-18-48-30-926074.pkl"
)

alll = [standard_algos, optimal_algos_30, optimal_algos_150]

idx = 0

t = pd.DataFrame(columns=["method", "target", "omic", "algo", "perf", "params"])

for i, this in enumerate(alll):
    targets = this.keys()
    for target in targets:
        print(target)
        omics = this[target].keys()
        for omic in omics:
            algos = this[target][omic].keys()
            for algo in algos:
                r = this[target][omic][algo]["result"]
                if type(r) is np.float64:
                    perf = r
                    params = 0
                elif type(r) is dict:
                    perf = r["target"]
                    params = r["params"]
                else:
                    perf = np.nan
                    params = np.nan
                t.loc[idx, "method"] = ["standard", "optimized_30", "optimized_150"][i]
                t.loc[idx, "target"] = target.split("_")[0]
                t.loc[idx, "omic"] = omic
                t.loc[idx, "algo"] = algo
                t.loc[idx, "perf"] = perf
                if i == 0:
                    t.loc[idx, "params"] = params
                else:
                    if type(params) is dict:
                        for param in params.keys():
                            t.loc[idx, param] = params[param]
                idx = idx + 1


g = sns.catplot(
    x="algo",
    y="perf",
    hue="method",
    row="target",
    col="omic",
    data=t,
    kind="bar",
    height=5,
    legend_out=True,
)
for ax in g.axes.flat:
    ax.set_ylim(0.5, 0.95)
plt.tight_layout()


#%%

# Calculating improvements

targets = t["target"].unique()
for target in targets:
    for omic in omics:
        for algo in algos:
            x = t.loc[
                (t["target"] == target) & (t["omic"] == omic) & (t["algo"] == algo),
                ["method", "perf"],
            ]
            imp1 = (
                x.loc[x["method"] == "optimized_30", "perf"].values
                - x.loc[x["method"] == "standard", "perf"].values
            )
            imp2 = (
                x.loc[x["method"] == "optimized_150", "perf"].values
                - x.loc[x["method"] == "optimized_30", "perf"].values
            )
            t.loc[
                (t["target"] == target)
                & (t["omic"] == omic)
                & (t["algo"] == algo)
                & (t["method"] == "optimized_30"),
                ["improvement_1"],
            ] = imp1
            t.loc[
                (t["target"] == target)
                & (t["omic"] == omic)
                & (t["algo"] == algo)
                & (t["method"] == "optimized_150"),
                ["improvement_2"],
            ] = imp2

fig = plt.figure()
sns.distplot(t["improvement_1"], rug=True, color="black")
t["improvement_1"].mean()

fig2 = plt.figure()
sns.distplot(t["improvement_2"], rug=True, color="black")
t["improvement_2"].mean()

t["improvement"] = t[["improvement_1", "improvement_2"]].sum(axis=1, min_count=1)

f, ax = plt.subplots(figsize=(15, 10))
sns.swarmplot(
    x="algo",
    y="improvement",
    data=t.loc[t["method"] == "optimized_30", :],
    color="black",
    ax=ax,
)
plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(15, 10))
sns.swarmplot(
    x="algo",
    y="improvement",
    data=t.loc[t["method"] == "optimized_150", :],
    color="black",
    ax=ax,
)
plt.xticks(rotation=90)

f, ax = plt.subplots(figsize=(15, 10))
sns.swarmplot(
    x="omic",
    y="improvement",
    data=t.loc[t["method"] == "optimized_30", :],
    color="black",
    ax=ax,
)
plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(15, 10))
sns.swarmplot(
    x="omic",
    y="improvement",
    data=t.loc[t["method"] == "optimized_150", :],
    color="black",
    ax=ax,
)
plt.xticks(rotation=90)

f, ax = plt.subplots(figsize=(15, 10))
sns.swarmplot(
    x="target",
    y="improvement",
    data=t.loc[t["method"] == "optimized_30", :],
    color="black",
    ax=ax,
)
plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(15, 10))
sns.swarmplot(
    x="target",
    y="improvement",
    data=t.loc[t["method"] == "optimized_150", :],
    color="black",
    ax=ax,
)
plt.xticks(rotation=90)

# TODO: end chunk 4
#%%
# TODO: start chunk 5
# get final_data from chunk 1
# get algos_dict from  chunk 2
# retrieve important features

all_omic = final_data.omic.unique().tolist()
all_omic.remove("DRUGS")

global_importances = {}

for target in algos_dict.keys():
    for omic in algos_dict[target].keys():
        xx = algos_dict[target][omic][0]
        fig, ax = plt.subplots(1, 3, figsize=(40, 12))
        axid = 0

        if omic == "complete":
            X = final_data.extract(omics_list=all_omic).to_pandas()
        else:
            X = final_data.to_pandas(omic=omic)

        y = final_data.to_pandas(omic="DRUGS").loc[:, target]
        to_keep = y[y != 0.5].index
        X = X.loc[to_keep, :]
        y = y[to_keep]

        all_importances = pd.DataFrame(index=X.columns)

        for idx in xx.index:
            isSVC = False
            isMLP = False
            model = xx[idx]
            print(f"{target} : {omic}: ")
            print(model)

            trained = model.fit(X, y)
            model_name = type(model).__name__
            print("model trained")
            try:
                importances = trained.coef_  # linear
                fi = pd.Series(data=importances[0], index=X.columns, name=model_name)
            except:
                try:
                    importances = trained.feature_importances_  # trees
                    fi = pd.Series(data=importances, index=X.columns, name=model_name)
                except:
                    try:
                        importances = trained._dual_coef_  # SVC?
                        isSVC = True
                        fi = pd.Series(
                            data=importances[0], index=X.columns, name=model_name
                        )
                    except:
                        try:
                            if not isSVC:
                                coeffs = trained.coefs_  # MLP
                                isMLP = True
                                # fi = pd.Series(data=importances, index=X.columns, name=model_name)
                        except:
                            raise ValueError(f"Did not recognize this model: {model}")
            if not (isSVC or isMLP):
                fi = fi.abs().sort_values(ascending=False)
                fir = fi[:9]
                print("plotting")
                sns.barplot(fir.index, fir.values, ax=ax[axid], palette="GnBu_d")
                ax[axid].set_xticklabels(fir.index, rotation=90, ha="right")
                ax[axid].set_title(model_name)
                all_importances[model_name] = fi
                axid = axid + 1
            else:
                ax[axid].set_title(model_name)

                # plt.xticks(rotation=90)
                # plt.title(model_name)
        fig.suptitle(target.split("_")[0] + " - " + omic)

        # figx, axx = plt.subplots(figsize=(12,12))
        n_imp = (all_importances - all_importances.min()) / (
            all_importances.max() - all_importances.min()
        )
        s = n_imp.sum(axis=1)
        n_imp = n_imp.loc[s.sort_values(ascending=False).index, :]
        toplot = n_imp.iloc[:50, :]
        sns.lineplot(data=toplot, dashes=False, sort=False, ax=ax[-1])
        ax[-1].set_xticklabels(toplot.index, rotation=90, ha="right")
        if n_imp.shape[1] > 1:
            c = n_imp.iloc[:, 0].corr(n_imp.iloc[:, 1], method="spearman")
        else:
            c = 0
        ax[-1].set_title("normalized importances - rho= " + str(round(c, 2)))
        if target not in global_importances.keys():
            global_importances[target] = {}

        global_importances[target][omic] = n_imp

# define outputs plots and tables not used by subsequent scripts list of most important features
# select top x, examine, needs other result, tables. Gene set enrichment.
#%%


genelist = [
    "IGKV4-1",
    "CD55",
    "IGKC",
    "PPFIBP1",
    "ABHD4",
    "PCSK6",
    "PGD",
    "ARHGDIB",
    "ITGB2",
    "CARD6",
    "MDM2",
    "TP53",
]

genesets = ["KEGG_2019_Human", "KEGG_2021_Human", "KEGG_2013", "KEGG_2016"]

enr = gsea.get_enrichr(genelist, genesets, cutoff=0.1, tag="test")


#%%
# for each target

# show ActArea vs IC50

# find extremes

# show doseresponses
