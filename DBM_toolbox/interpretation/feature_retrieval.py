from sklearn.inspection import permutation_importance
import logging
import pandas as pd
import numpy as np
import shap

def shuffle_feature(dataframe, column_name, random_state=42): #TODO: set up rng with random state for reproducibility
    df_shuffled = dataframe.copy()
    this_feature = df_shuffled.loc[:, column_name].values
    np.random.shuffle(this_feature)

    return df_shuffled


def permutation_importances(trained_model, predictors, target, folds=10, random_state=42):
    importances_df = pd.DataFrame(index=list(range(folds)), columns=predictors.columns)
    predictions_0 = trained_model.predict(predictors)
    perf_0 = np.mean(np.mean([predictions_0[x] == target[x] for x in range(len(target))]))
    for feature in predictors.columns:
        print(f"feature: {feature}", end="")
        for fold in range(folds):
            shuffled = shuffle_feature(dataframe=predictors, column_name=feature, random_state=random_state)
            predictions = trained_model.predict(shuffled)
            perf = np.mean(np.mean([predictions[x] == target[x] for x in range(len(target))]))
            print(f"{perf}")
            importances_df.loc[fold, feature] = perf_0 - perf

    return importances_df.mean()


def explain_model(model, predictors, target, folds=10, seed=42):
    logging.info(f"...explaining model {model}...")
    index1 = target.index[target.apply(np.isnan)]
    index2 = predictors.index[predictors.apply(np.isnan).any(axis=1)]
    indices_to_drop = index1.union(index2)
    n_dropped = len(indices_to_drop)
    npos = sum(target == 1)
    nneg = sum(target == 0)
    predictors = predictors.drop(indices_to_drop)
    target = target.drop(indices_to_drop)
    logging.info(
        f"X: {predictors.shape[0]} samples and {predictors.shape[1]} features"
    )
    logging.info(f"y: {target.size} samples, with {npos} positives and {nneg} negatives ({n_dropped} dropped)")

    trained_model = model.fit(predictors, target)
    importances_df = permutation_importances(trained_model, predictors, target, folds, random_state=seed)

    return importances_df


def shap_model(model, predictors, target, folds=10, seed=42):
    logging.info(f"...explaining model {model}...")
    index1 = target.index[target.apply(np.isnan)]
    index2 = predictors.index[predictors.apply(np.isnan).any(axis=1)]
    indices_to_drop = index1.union(index2)
    n_dropped = len(indices_to_drop)
    npos = sum(target == 1)
    nneg = sum(target == 0)
    predictors = predictors.drop(indices_to_drop)
    target = target.drop(indices_to_drop)
    logging.info(
        f"X: {predictors.shape[0]} samples and {predictors.shape[1]} features"
    )
    logging.info(f"y: {target.size} samples, with {npos} positives and {nneg} negatives ({n_dropped} dropped)")
    trained_model = model.fit(predictors, target)
    explainer = shap.TreeExplainer(trained_model)
    shap_values_model = explainer.shap_values(predictors)

    return shap_values_model[1]


def explain_all(models, predictors, target, folds=10, seed=42):
    logging.info("...starting model explanations...")
    models_list = list(models.keys())
    explanations = pd.DataFrame(index=predictors.columns, columns=models_list)
    for this_model_name in models_list:
        print(f"model: {this_model_name}")
        model = models[this_model_name]['estimator']
        explained_model = explain_model(model, predictors, target, folds, seed)
        explanations.loc[:, this_model_name] = explained_model.values
    return explanations


def shap_all(models, predictors, target, folds=10, seed=42):
    logging.info("...starting shap values calculations...")
    models_list = list(models.keys())
    shap_values = pd.DataFrame(index=predictors.columns, columns=models_list)
    for this_model_name in models_list:
        if this_model_name not in ["SVC", "SVM", "SVP", "Logistic", "Ridge", "EN", "KNN"]:
            print(f"model: {this_model_name}")
            model = models[this_model_name]['estimator']
            shaps = shap_model(model, predictors, target, folds, seed)
            shap_means = pd.DataFrame(data=shaps, columns=predictors.columns).mean(axis=0)
            shap_values.loc[:, this_model_name] = shap_means
        else:
            shap_values.loc[:, this_model_name] = 0
    return shap_values
