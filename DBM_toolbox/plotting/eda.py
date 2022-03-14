# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:52:06 2020

@author: sebde
"""
import os

from matplotlib import pyplot as plt
import matplotlib.colors as c
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import random

from DBM_toolbox.data_manipulation import preprocessing, dataset_class

# import missingno as msno


def doublesort(dataframe, ascending=True):
    dataframe["samplecompleteness"] = np.mean(dataframe, axis=1)
    dataframe = dataframe.append(
        pd.Series(data=np.mean(dataframe, axis=0), name="featurecompleteness")
    )
    semisorted_dataframe = dataframe.sort_values(
        by="samplecompleteness", ascending=True, axis=0
    )
    sorted_indices = semisorted_dataframe["samplecompleteness"]
    semisorted_dataframe = semisorted_dataframe.drop("samplecompleteness", axis=1)
    sorted_dataframe = semisorted_dataframe.sort_values(
        by="featurecompleteness", ascending=True, axis=1
    )
    sorted_columns = sorted_dataframe.loc["featurecompleteness", :]
    sorted_dataframe = sorted_dataframe.drop("featurecompleteness")

    return sorted_dataframe, sorted_columns, sorted_indices


def plot_eda_all(dataframe, title=None):
    """get plots for general data exploration"""

    ts = str(round(datetime.datetime.now().timestamp()))

    #     plot_eda_PCA(dataframe = dataframe, title=title, ts=ts)

    plot_eda_generaldistrib(dataframe=dataframe, title=title, ts=ts)

    plot_eda_meanvariance(dataframe=dataframe, title=title, ts=ts)

    plot_eda_missingsummary(dataframe=dataframe, title=title, ts=ts)

    plot_eda_correl(dataframe=dataframe, title=title, ts=ts)

    plot_eda_missingcorrel(dataframe=dataframe, title=title, ts=ts)


def plot_overlaps(
    dataset, title, outputdir=None
):  # TODO: complete function to display venn diagrams of database samples
    omics = list(set(dataset.omic))
    databases = list(set(dataset.database))
    dataframe = dataset.dataframe

    res = dict()

    for database in databases:
        for omic in omics:
            dataframe = dataset.to_pandas(omic=omic, database=database)
            if dataframe.shape[1] > 0:
                dataframe = dataframe.dropna()
                data_name = omic + "/" + database
                res[data_name] = list(dataframe.index)
                pass

            # except:
            #     pass


def plot_eda_PCA(dataframe, title, ts):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(dataframe.dropna())
    principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

    #     final_df = pd.concat([principal_df, df[['target']]], axis = 1)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.scatterplot(x="PC1", y="PC2", data=principal_df)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)


def plot_eda_generaldistrib(dataframe, title, ts):

    print(f"general distribution plot for {title}...")

    ncol = dataframe.shape[1]
    if ncol > 100:
        dataframe = dataframe.iloc[:, random.sample(range(dataframe.shape[1]), 100)]
        title = title + "_sample"
    try:
        fig, axes = plt.subplots(2, 1, figsize=(25, 10), sharex=True)
        sns.set_context("talk")
        distr = sns.stripplot(
            data=dataframe, jitter=True, color="k", size=3, ax=axes[0]
        )
        distr.set_xticklabels(distr.get_xticklabels(), rotation=90)
        distr.set_title("raw")
        distr2 = sns.stripplot(
            data=np.log10(dataframe), jitter=True, color="k", size=3, ax=axes[1]
        )
        distr2.set_xticklabels(distr2.get_xticklabels(), rotation=90)
        distr2.set_title("log")
        fig.suptitle(title)

        plt.savefig(ts + "_" + title + "_distrib.svg")
    except:
        print("no stripplot")


def plot_eda_meanvariance(dataframe, title, ts):

    print(f"mean-variance plot for {title}...")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        sns.set_context("talk")
        means = dataframe.mean().rename("Mean")
        stds = dataframe.std().rename("Std")
        toplot = pd.concat([means, stds], axis=1)
        mv = sns.scatterplot(x="Mean", y="Std", data=toplot, color="k", ax=axes[0])
        mv.set_title("raw")
        means = (np.log10(dataframe)).mean().rename("Mean")
        stds = (np.log10(dataframe)).std().rename("Std")
        toplot = pd.concat([means, stds], axis=1).dropna()
        mv2 = sns.scatterplot(x="Mean", y="Std", data=toplot, color="k", ax=axes[1])
        mv2.set_title("log")
        fig.suptitle(title)

        plt.savefig(ts + "_" + title + "_mean-sd.svg")
    except:
        print("no mean-variance plot")


def plot_eda_missingsummary(dataframe, title, ts):

    print(f"missing data plot for {title}...")

    try:
        bool_df = ~dataframe.isna()

        bool_df, sorted_featcompl, sorted_samplecompl = doublesort(bool_df)

        fig = plt.figure(figsize=(22, 22))
        sns.set_context("talk")
        gs = GridSpec(5, 5)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        #         ax3 = fig.add_subplot(gs[2:,1:])

        line1 = sns.lineplot(data=sorted_featcompl.to_numpy()[::-1], ax=ax1)
        line1.set_ylabel("feature completeness")

        line2 = sns.lineplot(data=sorted_samplecompl.to_numpy()[::-1], ax=ax2)
        line2.set_ylabel("sample completeness")

    #         colors = {'black':1, 'white':0}
    #         cMap = c.ListedColormap(colors)
    #         miss = sns.heatmap(data=bool_df, cbar=False, cmap=cMap, ax=ax3)
    #         fig.suptitle(title)
    #         plt.savefig(ts + '_' +  title + '_missing.svg')
    except:
        print("no missing data plot")


def plot_eda_correl(dataframe, title, ts):

    print(f"correlation plot for {title}...")

    ncol = dataframe.shape[1]
    if ncol > 5000:
        dataframe = dataframe.iloc[:, random.sample(range(dataframe.shape[1]), 5000)]
        title = title + "_sample"

    try:
        featcorrel = dataframe.corr()
        sorted_featcorrel = np.abs(doublesort(featcorrel)[0])
        samplecorrel = dataframe.transpose().corr()
        sorted_samplecorrel = np.abs(doublesort(samplecorrel)[0])

        fig = plt.figure(figsize=(22, 22))
        sns.set_context("talk")
        gs = GridSpec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        #         ax3 = fig.add_subplot(gs[1:, 0])
        #         ax4 = fig.add_subplot(gs[1:, 1])

        line1 = sns.distplot(
            sorted_featcorrel.to_numpy().flatten(), norm_hist=True, ax=ax1
        )
        line1.set_title("feature correlations")
        line2 = sns.distplot(
            sorted_samplecorrel.to_numpy().flatten(), norm_hist=True, ax=ax2
        )
        line2.set_title("sample correlations")

    #         corr1 = sns.heatmap(data=sorted_featcorrel, cbar=False, cmap='magma_r', ax=ax3)
    #         corr1.set_title('feature correlation matrix')
    #         corr2 = sns.heatmap(data=sorted_samplecorrel, cbar=False, cmap='magma_r', ax=ax4)
    #         corr2.set_title('sample correlation matrix')
    #         fig.suptitle(title)
    #         plt.savefig(ts + '_' +  title + '_correl.svg')
    except:
        print("no correlation plot")


def plot_eda_missingcorrel(dataframe, title, ts):

    print(f"missing data correlation plot for {title}...")

    ncol = dataframe.shape[1]
    if ncol > 5000:
        dataframe = dataframe.iloc[:, random.sample(range(dataframe.shape[1]), 5000)]
        title = title + "_sample"

    try:
        isdata = ~dataframe.isna()
        miss_featcorrel = isdata.astype(int).corr()
        sorted_missfeatcorrel = np.abs(doublesort(miss_featcorrel)[0])
        miss_samplecorrel = isdata.transpose().corr()
        sorted_misssamplecorrel = np.abs(doublesort(miss_samplecorrel)[0])

        fig = plt.figure(figsize=(22, 22))
        sns.set_context("talk")
        gs = GridSpec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        #         ax3 = fig.add_subplot(gs[1:, 0])
        #         ax4 = fig.add_subplot(gs[1:, 1])

        line1 = sns.distplot(
            sorted_missfeatcorrel.to_numpy().flatten(), norm_hist=True, ax=ax1
        )
        line1.set_title("missing feature correlations")
        line2 = sns.distplot(
            sorted_misssamplecorrel.to_numpy().flatten(), norm_hist=True, ax=ax2
        )
        line2.set_title("missing sample correlations")

    #         corr1 = sns.heatmap(data=sorted_missfeatcorrel, cbar=False, cmap='magma_r', ax=ax3)
    #         corr1.set_title('missing feature correlation matrix')
    #         corr2 = sns.heatmap(data=sorted_misssamplecorrel, cbar=False, cmap='magma_r', ax=ax4)
    #         corr2.set_title('missing sample correlation matrix')
    #         fig.suptitle(title)
    #         plt.savefig(ts + '_' + title + '_missingcorrel.svg')
    except:
        print("no missing data correlation plot")


def plot_target(dataframe, ActAreas, IC50s, dr, bounds, outputdir=None):
    title = dataframe.name
    ts = str(round(datetime.datetime.now().timestamp()))
    fig, ax = plt.subplots(2, 1, figsize=(15, 15))
    sns.distplot(dataframe, bins=50, rug=True, ax=ax[0])
    df = preprocessing.rescale_data(dataframe)
    points = sns.distplot(df, hist=False, ax=ax[1]).get_lines()[0].get_data()
    x = points[0]
    y = points[1]

    q = np.quantile(df.dropna(), bounds)

    ax[1].fill_between(x, y, where=x >= q[1], color="g")
    ax[1].fill_between(x, y, where=x <= q[0], color="r")
    ax[1].fill_between(x, y, where=(x <= q[1]) & (x >= q[0]), color="y")
    fig.suptitle(title)
    if outputdir:
        out_path = os.path.join(outputdir, ts + "_" + title + "_distr.svg")
    else:
        out_path = ts + "_" + title + "_distr.svg"
    plt.savefig(out_path)

    labels = dataset_class.Dataset(
        dataframe=dataframe.to_frame(), omic="DRUGS", database="mod"
    )
    labels = labels.data_pop_quantize(target_omic="DRUGS", quantiles_df=bounds)
    # TODO: check location of plot
    plot_dose_response(dr, target=title.split("_")[0], labels=labels.dataframe)

    ##
    # TODO: check location of plot
    plot_scatter_dr(dataframe, ActAreas, IC50s, dr, bounds, labels)


def plot_modeling_results(dataframe, outputdir=None):
    ts = str(round(datetime.datetime.now().timestamp()))
    targets = list(set(dataframe["target"]))
    omics = list(set(dataframe["omic"]))
    algos = list(set(dataframe["algo"]))
    palette = "colorblind"
    linewidth = 1.5
    capsize = 0.1
    edgecolor = ".2"
    for this_target in targets:
        plt.figure(figsize=(15, 15))
        ax = sns.barplot(
            x="algo",
            y="perf",
            hue="omic",
            palette=palette,
            linewidth=linewidth,
            capsize=capsize,
            edgecolor=edgecolor,
            ci=None,
            data=dataframe[dataframe["target"] == this_target],
        ).set_title(this_target)
        plt.xticks(rotation=90)
        if outputdir:
            out_path = os.path.join(outputdir, ts + "_" + this_target + "_.svg")
        else:
            out_path = ts + "_" + this_target + "_.svg"
        plt.savefig(out_path)
    for this_omic in omics:
        plt.figure(figsize=(15, 15))
        ax = sns.barplot(
            x="target",
            y="perf",
            hue="algo",
            palette=palette,
            linewidth=linewidth,
            capsize=capsize,
            edgecolor=edgecolor,
            ci=None,
            data=dataframe[dataframe["omic"] == this_omic],
        ).set_title(this_omic)
        plt.xticks(rotation=90)
        if outputdir:
            out_path = os.path.join(outputdir, ts + "_" + this_omic + "_.svg")
        else:
            out_path = ts + "_" + this_omic + "_.svg"
        plt.savefig(out_path)
    for this_algo in algos:
        plt.figure(figsize=(15, 15))
        ax = sns.barplot(
            x="target",
            y="perf",
            hue="omic",
            palette=palette,
            linewidth=linewidth,
            capsize=capsize,
            edgecolor=edgecolor,
            ci=None,
            data=dataframe[dataframe["algo"] == this_algo],
        ).set_title(this_algo)
        plt.xticks(rotation=90)
        if outputdir:
            out_path = os.path.join(outputdir, ts + "_" + this_algo + "_.svg")
        else:
            out_path = ts + "_" + this_algo + "_.svg"
        plt.savefig(out_path)
    for this_target in targets:
        plt.figure(figsize=(15, 15))
        ax = sns.barplot(
            x="algo",
            y="perf",
            palette=palette,
            linewidth=linewidth,
            capsize=capsize,
            edgecolor=edgecolor,
            ci=None,
            data=dataframe[dataframe["target"] == this_target],
        ).set_title(this_target)
        plt.xticks(rotation=90)
        if outputdir:
            out_path = os.path.join(outputdir, ts + "_" + this_target + "_2.svg")
        else:
            out_path = ts + "_" + this_target + "_2.svg"
        plt.savefig(out_path)
    for this_omic in omics:
        plt.figure(figsize=(15, 15))
        ax = sns.barplot(
            x="target",
            y="perf",
            palette=palette,
            linewidth=linewidth,
            capsize=capsize,
            edgecolor=edgecolor,
            ci=None,
            data=dataframe[dataframe["omic"] == this_omic],
        ).set_title(this_omic)
        plt.xticks(rotation=90)
        if outputdir:
            out_path = os.path.join(outputdir, ts + "_" + this_omic + "_2.svg")
        else:
            out_path = ts + "_" + this_omic + "_2.svg"
        plt.savefig(out_path)
    for this_algo in algos:
        plt.figure(figsize=(15, 15))
        ax = sns.barplot(
            x="omic",
            y="perf",
            palette=palette,
            linewidth=linewidth,
            capsize=capsize,
            edgecolor=edgecolor,
            ci=None,
            data=dataframe[dataframe["algo"] == this_algo],
        ).set_title(this_algo)
        plt.xticks(rotation=90)
        if outputdir:
            out_path = os.path.join(outputdir, ts + "_" + this_algo + "_2.svg")
        else:
            out_path = ts + "_" + this_algo + "_2.svg"
        plt.savefig(out_path)

    try:
        fig, ax = plt.figure(figsize=(15, 15))
    except TypeError:
        pass

    # TODO: this part bugs, needs to produce the plts and save them without showing them as it comsumes too much memory...

    # sns.scatterplot(x='perf', y='N', hue='target', style='algo', data=dataframe, ax=ax)
    # if outputdir:
    #     out_path = os.path.join(outputdir, ts + '_overall.svg')
    # else:
    #     out_path = ts + '_overall.svg'
    # plt.savefig(out_path)


def plot_dose_response(dose_responses, idxs=None, target=None, labels=None):

    if idxs is None:
        idxs = dose_responses.index
    if target is None:
        target = dose_responses.columns
    if labels is None:
        labels = dose_responses
        labels.loc[:, :] = 0.5
    if type(idxs) is str:
        idxs = [idxs]

    cols = [x for x in dose_responses.columns if x.startswith(target.split("_")[0])]
    df = dose_responses.loc[:, cols]
    df = df.reindex(idxs)

    cols = [x for x in labels.columns if x.startswith(target.split("_")[0])]
    labels = labels.loc[:, cols]
    labels = labels.reindex(idxs)

    fig, ax = plt.subplots(figsize=(15, 15))
    dfx = df  # .loc[:, cols]
    labelx = labels  # .loc[:, col]

    for n_idx, idx in enumerate(idxs):
        col = [x for x in labelx if x.startswith(target)]
        this_val = labelx.loc[idx, col].values[0]

        print(idx)
        if not np.isnan(this_val):
            if this_val == 1:
                color = "green"
            elif this_val == 0:
                color = "red"
            else:
                color = "grey"
            try:
                sns.lineplot(
                    x=np.log(dfx.iloc[n_idx, 0]), y=dfx.iloc[n_idx, 1], color=color
                )
                plt.title(target)  # TODO: save plot in correct folder outputdir
            except:
                print("could not display curve")


def plot_scatter_dr(dataframe, ActAreas, IC50s, dr, bounds, labels):

    fig, ax = plt.subplots(figsize=(10, 10))
    if isinstance(dataframe, pd.Series):
        dataframe = dataframe.to_frame()
    for col in dataframe.columns:
        target = col.split("_")[0]
        xval = ActAreas.loc[
            :, [x for x in ActAreas.columns.tolist() if x.startswith(target)]
        ]
        yval = IC50s.loc[:, [y for y in IC50s.columns.tolist() if y.startswith(target)]]

        max_val = xval.max()
        min_val = xval.min()
        bound0 = bounds[0] * (max_val - min_val) + min_val
        bound1 = bounds[1] * (max_val - min_val) + min_val
        xs_mid = xval > bound0
        xs_high = xval > bound1
        xs = xs_mid.astype(int) + xs_high.astype(int)
        for _ in xs.index:
            print(xs.loc[_, :])
        xs = xs.rename(columns={xs.columns[0]: "id"})
        xs[xs == 0] = "R"
        xs[xs == 1] = "I"
        xs[xs == 2] = "S"
        df = pd.merge(xval, yval, left_index=True, right_index=True)
        df = pd.merge(df, xs, left_index=True, right_index=True)
        df = df.sort_values(by="id", ascending=False)
        sns.scatterplot(
            data=df,
            x=df.columns[0],
            y=df.columns[1],
            hue=df.columns[2],
            palette=["red", "yellow", "green"],
            ax=ax,
        )
        # TODO: save plot in correct outputdir
        plt.title(target)
        print(df)
