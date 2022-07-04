from sklearn.inspection import permutation_importance
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


def shuffle_feature(dataframe, column_name, random_state=42): #TODO: set up rng with random state for reproducibility
    df_shuffled = dataframe.copy()
    this_feature = df_shuffled.loc[:, column_name].values
    np.random.shuffle(this_feature)

    return df_shuffled


def permutation_essentialities(model, predictors, target, folds=5, random_state=42):
    importances_df = pd.DataFrame(index=list(range(folds)), columns=predictors.columns)
    xval = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    predictions_0 = cross_val_predict(model, predictors, target, cv=xval, n_jobs=-1)
    perf_0 = np.mean(np.mean([predictions_0[x] == target[x] for x in range(len(target))]))
    for feature in predictors.columns:
        print(f"feature: {feature} with perf {perf_0}... ", end="")
        for fold in range(10):
            shuffled = shuffle_feature(dataframe=predictors, column_name=feature, random_state=random_state)

            predictions = cross_val_predict(model, shuffled, target, cv=xval, n_jobs=-1)
            perf = np.mean(np.mean([predictions[x] == target[x] for x in range(len(target))]))
            print(f"{perf}, ", end="")
            importances_df.loc[fold, feature] = perf_0 - perf
        print("")
    return importances_df.mean()


def permutation_importances(model, predictors, target, folds=5, random_state=42):
    importances_df = pd.DataFrame(index=list(range(folds)), columns=predictors.columns)
    train_predictions = model.predict(predictors)
    train_perf = np.mean(np.mean([train_predictions[x] == target[x] for x in range(len(target))]))
    for feature in predictors.columns:
        print(f"feature: {feature}... ", end="")
        for fold in range(folds):
            shuffled = shuffle_feature(dataframe=predictors, column_name=feature, random_state=random_state)
            shuffled_predictions = model.predict(shuffled)
            shuffled_perf = np.mean(np.mean([shuffled_predictions[x] == target[x] for x in range(len(target))]))
            print(f"{train_perf - shuffled_perf}, ", end="")
            importances_df.loc[fold, feature] = train_perf - shuffled_perf
        print("")
    return importances_df.mean()


def explain_model(model, predictors, target, folds=5, seed=42):
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


def explain_all(models, predictors, target, folds=5, seed=42):
    logging.info("...starting model explanations...")
    models_list = list(models.keys())
    explanations = pd.DataFrame(index=predictors.columns, columns=models_list)
    for this_model_name in models_list:
        print(f"model: {this_model_name}")
        model = models[this_model_name]['estimator']
        model = model.fit(predictors, target)
        if hasattr(model, "feature_importances_"):
            # TODO: implement this but as the average of 5-fold cv
            explained_model = model.feature_importances_
        else:
            explained_model = explain_model(model, predictors, target, folds, seed)
            explained_model = explained_model.values
        explanations.loc[:, this_model_name] = explained_model
    return explanations

