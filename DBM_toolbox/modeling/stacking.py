# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:01:57 2020

@author: sebde
"""
import numpy as np
import pandas as pd
from vecstack import stacking
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


def stack_models(
    models,
    final_model,
    X_train,
    y_train,
    X_test,
    y_test,
    folds,
    metric=roc_auc_score,
    seed=42,
):
    """Constructs, fits, predicts using XGBoost on a bag of models,
    using only the models' predictions as input
    """
    print("stacking models...")
    S_train, S_test = stacking(
        models,
        X_train,
        y_train,
        X_test,
        regression=False,
        mode="oof_pred_bag",
        needs_proba=False,
        save_dir=None,
        metric=metric,
        n_folds=folds,
        stratified=True,
        shuffle=True,
        random_state=seed,
        verbose=0,
    )
    model = final_model
    print("fitting stack...")
    model = model.fit(S_train, y_train)
    y_pred_train = model.predict(S_train)
    y_pred = model.predict(S_test)
    y_proba = model.predict_proba(S_test)
    train_AUC = roc_auc_score(y_train, y_pred_train)
    finalAUC = roc_auc_score(y_test, y_pred)
    return model, finalAUC, train_AUC, y_pred, y_proba


def stack_extended_models(
    models,
    final_model,
    X_train,
    y_train,
    X_test,
    y_test,
    folds,
    metric=roc_auc_score,
    seed=42,
):
    """Constructs, fits, predicts using XGBoost on a bag of models,
    using all the training data plus the other models' predictions as input
    """
    S_train, S_test = stacking(
        models,
        X_train,
        y_train,
        X_test,
        regression=False,
        mode="oof_pred_bag",
        needs_proba=False,
        save_dir=None,
        metric=metric,
        n_folds=folds,
        stratified=True,
        shuffle=True,
        random_state=seed,
        verbose=0,
    )
    model = final_model
    E_train = np.concatenate([S_train, X_train], axis=1)
    E_test = np.concatenate([S_test, X_test], axis=1)
    model = model.fit(E_train, y_train)
    y_pred = model.predict(E_test)
    y_proba = model.predict_proba(E_test)
    finalAUC = roc_auc_score(y_test, y_pred)
    print("Final AUC: [%.8f]" % finalAUC)
    return model, finalAUC, y_pred, y_proba


def compute_stacks(
    dataset, models, final_model, targets_list, metric="roc_auc", folds=10, seed=42
):
    best_stacks = dict()
    for target in targets_list:
        print(f"Computing stack for {target}")
        this_dataset = dataset.to_binary(target=target)
        y = this_dataset.to_pandas()[target]
        omics = models[target]
        predictions = pd.DataFrame(index=y.index)
        scores = pd.Series(index=["full", "lean"])
        importances = pd.DataFrame()
        for omic in omics.keys():
            if omic == "complete":
                X = this_dataset.to_pandas().drop(targets_list, axis=1)
            else:
                X = this_dataset.to_pandas(omic=omic)
                print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
                print(f"y: {y.size} samples")
                X = X.dropna(how="all")
                print("dropping")
                print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
                print(f"y: {y.size} samples")
                index1 = y.index[
                    y.apply(np.isnan)
                ]
                index2 = X.index[X.apply(np.isnan).any(axis=1)]
                indices_to_drop = index1.union(index2)
                print(f"cross-dropping: idx1: {index1}, idx2: {index2}")
                X = X.drop(indices_to_drop)
                y = y.drop(indices_to_drop)
                print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
                print(f"y: {y.size} samples")

            models_list = omics[omic]
            for idx, model in enumerate(models_list):

                this_model = model.iloc[0]

                xval = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                print(X, y, xval)
                omic_predict = cross_val_predict(this_model, X, y, cv=xval, n_jobs=-1)
                feature_name = omic + str(idx)
                predictions[feature_name] = omic_predict

        data_with_predictions = (
            this_dataset.to_pandas()
            .drop(targets_list, axis=1)
            .merge(predictions, left_index=True, right_index=True)
        )
        full_stack = final_model.fit(data_with_predictions, y, eval_metric="auc")
        full_stack_predict = cross_val_predict(
            final_model, data_with_predictions, y, cv=xval, n_jobs=-1
        )
        scores["full"] = np.mean(
            cross_val_score(
                final_model,
                data_with_predictions,
                y,
                scoring=metric,
                cv=xval,
                n_jobs=-1,
            )
        )

        lean_stack = final_model.fit(predictions, y, eval_metric="auc")
        lean_stack_predict = cross_val_predict(
            final_model, predictions, y, cv=xval, n_jobs=-1
        )
        scores["lean"] = np.mean(
            cross_val_score(
                final_model, predictions, y, scoring=metric, cv=xval, n_jobs=-1
            )
        )

        predictions["full_stack"] = full_stack_predict
        predictions["lean_stack"] = lean_stack_predict
        predictions["truth"] = y.values

        try:
            importances["full"] = pd.DataFrame(
                full_stack.feature_importances_, index=full_stack.feature_names
            )
            importances["lean"] = pd.DataFrame(
                lean_stack.feature_importances_, index=lean_stack.feature_names
            )
        except:
            importances = np.nan
        best_stacks[target] = {
            "support": importances,
            "scores": scores,
            "predictions": predictions,
        }
    return best_stacks

def compute_systematic_stacks(
    dataset, models, final_model, targets_list, metric="roc_auc", folds=10, seed=42
):
    res = {}
    for target in targets_list:
        this_dataset = dataset.to_binary(target=target)
        y = this_dataset.to_pandas()[target]
        predictions = pd.DataFrame(index=y.index)
        omics_dict = models[target]
        omics_list = omics_dict.keys()
        for this_omic in omics_list:
            if this_omic == "complete":
                X = this_dataset.to_pandas().drop(targets_list, axis=1)
                #print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
                #print(f"y: {y.size} samples")
                #X = X.dropna(how="all")
                #print("dropping")
             #   print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
             #   print(f"y: {y.size} samples")
             #   index1 = y.index[
             #       y.apply(np.isnan)
              #  ]  ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost
             #   index2 = X.index[X.apply(np.isnan).any(axis=1)]  ## SOLVED?
             #   indices_to_drop = index1.union(index2)
            #    print(f"cross-dropping: idx1: {index1}, idx2: {index2}")
            #    X = X.drop(indices_to_drop)
            #    y = y.drop(indices_to_drop)
            #    print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
            #    print(f"y: {y.size} samples")
            else:
                X = this_dataset.to_pandas(omic=this_omic)
                #print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
                #print(f"y: {y.size} samples")
                #X = X.dropna(how="all")
                #print("dropping")
            print(f"omic: {this_omic}, X: {X.shape[0]} samples and {X.shape[1]} features, y: {y.size} samples")
            index1 = y.index[
                y.apply(np.isnan)
            ]  ### TODO: this does not work as expected, if there are missing target values this is a problem for xgboost
            index2 = X.index[X.apply(np.isnan).any(axis=1)]  ## SOLVED?
            indices_to_drop = index1.union(index2)
            print(f"cross-dropping: idx1: {index1}, idx2: {index2}")
            try:
                X = X.drop(indices_to_drop)
            except:
                pass
            try:
                y = y.drop(indices_to_drop)
            except:
                pass
            print(f"omic: {this_omic}, X: {X.shape[0]} samples and {X.shape[1]} features, y: {y.size} samples")
            if X.shape[0] != y.size:
                print("intersecting:...")
                indexx = list(X.index)
                indexy = list(y.index)
                intersect = [value for value in indexx if value in indexy]
                X = X.loc[intersect, :]
                y = y.loc[intersect]
                print(f"omic: {this_omic}, X: {X.shape[0]} samples and {X.shape[1]} features, y: {y.size} samples")
            models_dict = omics_dict[this_omic]
            print(models_dict)
            models_list = list(models_dict.keys())
            print(f"models: {models_list}")
            for id, model in enumerate(models_list):
                this_model = models_dict[model]["estimator"]
                print("++++++++++++++++")
                print(this_model)
                print(f"omic: {this_omic}, X: {X.shape[0]} samples and {X.shape[1]} features, y: {y.size} samples")
                this_idx = y.index
                xval = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                omic_predict = cross_val_predict(this_model, X, y, cv=xval, n_jobs=-1)
                feature_name = this_omic + "_" + model
                predictions.loc[this_idx, feature_name] = omic_predict

        #stacking:
        matrix = pd.DataFrame(index=predictions.columns, columns=predictions.columns)
        l = len(predictions.columns)
        for pred1 in range(l-1):
            for pred2 in range(pred1, l):
                if pred1 == pred2:
                    merged_pred = predictions.iloc[:, [pred1]]
                else:
                    merged_pred = predictions.iloc[:, [pred1, pred2]]
                print(merged_pred)
                print(f"stacking {pred1} with {pred2}")
                print(f"omic: {this_omic}, X: {merged_pred.shape[0]} samples and {merged_pred.shape[1]} features, y: {y.size} samples")
                indexx = list(merged_pred.index)
                indexy = list(y.index)
                intersect = [value for value in indexx if value in indexy]
                merged_pred = merged_pred.loc[intersect, :]
                y = y.loc[intersect]
                print(f"omic: {this_omic}, X: {merged_pred.shape[0]} samples and {merged_pred.shape[1]} features, y: {y.size} samples")
                stack = final_model.fit(merged_pred, y, eval_metric="auc")
                perf = np.mean(
                    cross_val_score(
                        final_model, merged_pred, y, scoring=metric, cv=xval, n_jobs=-1
                    )
                )
                print(f"perf: {perf}")
                matrix.iloc[pred1, pred2] = perf
        res[target] = matrix
        print(matrix)
    return res






