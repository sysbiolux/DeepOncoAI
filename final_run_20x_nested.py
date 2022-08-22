
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

outer_folds = 20
inner_folds = 20

########################################
final_results = dict()
feature_importances = dict()
fprs = dict()
tprs = dict()
roc_aucs = dict()

for n in range(23):  # for each target
    idx = str(n+1)
    if len(idx) == 1:
        idx = '0' + idx

    final_data = unpickle_objects('f_testall_data_' + idx + '.pkl')

    # final_data = unpickle_objects('f_testmin_2_data_2022-08-04-13-30-00-019951.pkl')

    target_name = final_data.dataframe.columns[-1]
    final_data = final_data.to_binary(target=target_name)

    n_samples = len(final_data.dataframe.index)
    split_size = np.floor(n_samples/outer_folds)
    outer_mixed = rng.choice(final_data.dataframe.index, size=n_samples, replace=False)

    outer_train_idx = dict()
    outer_test_idx = dict()
    inner_train_idx = dict()
    inner_test_idx = dict()
    feature_importances[target_name] = dict()
    final_results[target_name] = pd.DataFrame(final_data.dataframe.iloc[:, -1].copy())  # truth

    trained_models = config.get_models(dataset=final_data, method="standard")

    for outer_loop in range(outer_folds):

        feature_importances[target_name][outer_loop] = dict()
        if outer_loop == outer_folds - 1:  # if it is the last split, put all remaining samples in
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

        first_level_preds = pd.DataFrame(final_data.dataframe.iloc[:, -1].copy())  # truth

        for inner_loop in range(inner_folds):
            print(f"target: {target_name} ({idx}), out: {outer_loop+1}/{outer_folds}, in: {inner_loop+1}/{inner_folds}")
            if inner_loop == inner_folds - 1:
                inner_test_idx[outer_loop][inner_loop] = inner_mixed[int(inner_loop * split_inner_size): -1]
            else:
                inner_test_idx[outer_loop][inner_loop] = inner_mixed[int(inner_loop * split_inner_size): int((inner_loop + 1) * split_inner_size)]
            inner_train_idx[outer_loop][inner_loop] = [x for x in inner_mixed if x not in inner_test_idx[outer_loop][inner_loop]]

            train_dataset, test_dataset = rest_dataset.split(train_index=inner_train_idx[outer_loop][inner_loop], test_index=inner_test_idx[outer_loop][inner_loop])

            for omic in trained_models[target_name].keys():  # for each omic type
                if omic != 'complete':
                    print(f'omic: {omic}')
                    train_features = train_dataset.to_pandas(omic=omic)
                    train_labels = train_dataset.to_pandas(omic='DRUGS').values.ravel()
                    test_features = test_dataset.to_pandas(omic=omic)
                    test_labels = test_dataset.to_pandas(omic='DRUGS').values.ravel()
                    for algo in trained_models[target_name][omic].keys():  # for each algo
                        print(f'algo: {algo}')
                        this_model = trained_models[target_name][omic][algo]['estimator']
                        this_model.fit(train_features, train_labels)
                        try:
                            this_predictions = this_model.predict_proba(test_features)
                            this_predictions = this_predictions[:, 1]
                        except:
                            this_predictions = this_model.predict(test_features)
                        first_level_preds.loc[inner_test_idx[outer_loop][inner_loop], 'pred_' + omic + '_' + algo] = this_predictions

        for omic in trained_models[target_name].keys():  # for each omic type
            if omic != 'complete':
                print(f'omic: {omic}')
                train_features = rest_dataset.to_pandas(omic=omic)
                train_labels = rest_dataset.to_pandas(omic='DRUGS').values.ravel()
                test_features = valid_dataset.to_pandas(omic=omic)
                test_labels = valid_dataset.to_pandas(omic='DRUGS').values.ravel()
                for algo in trained_models[target_name][omic].keys():  # for each algo
                    print(f'algo: {algo}')
                    this_model = trained_models[target_name][omic][algo]['estimator']
                    this_model.fit(train_features, train_labels)
                    try:
                        this_predictions = this_model.predict_proba(test_features)
                        this_predictions = this_predictions[:, 1]
                    except:
                        this_predictions = this_model.predict(test_features)
                    first_level_preds.loc[outer_test_idx[outer_loop], 'pred_' + omic + '_' + algo] = this_predictions

        cols = first_level_preds.columns
        train_features = first_level_preds.loc[outer_train_idx[outer_loop], cols[1:]]
        train_labels = first_level_preds.loc[outer_train_idx[outer_loop], cols[0]]
        test_features = first_level_preds.loc[outer_test_idx[outer_loop], cols[1:]]
        test_labels = first_level_preds.loc[outer_test_idx[outer_loop], cols[0]]
        train_features, train_labels = data_utils.merge_and_clean(train_features, train_labels)

        for algo in trained_models[target_name][omic].keys():
            this_model = trained_models[target_name][omic][algo]['estimator']
            this_model.fit(train_features, train_labels)
            try:
                this_predictions = this_model.predict_proba(test_features)
                this_predictions = this_predictions[:, 1]
            except:
                this_predictions = this_model.predict(test_features)
            try:
                f_imp = this_model.feature_importances_
            except:
                f_imp = np.nan
            feature_importances[target_name][outer_loop][algo] = pd.DataFrame(f_imp, index=this_model.feature_names_in_, columns=[algo])
            final_results[target_name].loc[outer_test_idx[outer_loop], 'pred2_' + algo] = this_predictions
            pickle_objects(final_results, "final_results.pkl")

    # draw ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    global_models = [x for x in final_results[target_name].columns if x.startswith('pred2')]
    plt.figure()
    for global_model in global_models:
        y_score = final_results[target_name].loc[:, global_model]
        y_score = y_score[y_score.isna() == False]
        y_true = final_results[target_name].loc[:, target_name]
        y_true = y_true.loc[y_score.index]
        fpr[global_model], tpr[global_model], _ = roc_curve(y_true, y_score)
        roc_auc[global_model] = auc(fpr[global_model], tpr[global_model])
        plt.plot(fpr[global_model], tpr[global_model], label=f"{global_model}: {round(roc_auc[global_model], 3)}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC for {target_name}")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f'ROC_{target_name}')

    fprs[target_name] = fpr
    tprs[target_name] = tpr
    roc_aucs[target_name] = roc_auc
    config.save(to_save=[fprs, tprs, roc_aucs], name="roc_info")
    config.save(to_save=final_results, name="final_results")
    config.save(to_save=feature_importances, name="features_imp")

### Analysis

all_features = pd.DataFrame()
for this_drug in feature_importances.keys():
    for split in feature_importances[this_drug].keys():
        my_features = feature_importances[this_drug][split]['RFC']
        if all_features.size < 1:
            all_features = my_features
        else:
            all_features = all_features.join(my_features, lsuffix='_x', rsuffix='_y')

all_features.to_csv('all_features.csv')


