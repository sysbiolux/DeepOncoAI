from sklearn.inspection import permutation_importance
import logging
import pandas as pd
import numpy as np

def explain_model(model, predictors, target, folds=5, seed=42): #fold is for use with CV (ELI5)
    logging.info(f"...explaining model {model}...")
    index1 = target.index[target.apply(np.isnan)]
    index2 = predictors.index[predictors.apply(np.isnan).any(axis=1)]
    indices_to_drop = index1.union(index2)
    # TODO: log number of dropped here
    n_dropped = len(indices_to_drop)
    npos = sum(target == 1)
    nneg = sum(target == 0)
    predictors = predictors.drop(indices_to_drop)
    target = target.drop(indices_to_drop)
    trained_model = model.fit(predictors, target)
    importances = permutation_importance(trained_model, predictors, target, random_state=seed, n_jobs=-1)
    importances_df = pd.Series(data=importances.importances_mean, index=predictors.columns)
    return importances_df


def explain_all(models, predictors, target, folds=5, seed=42): #fold is for use with CV (ELI5)
    logging.info("...starting model explanations...")
    models_list = list(models.keys())
    explanations = pd.DataFrame(index=predictors.columns, columns=models_list)
    for this_model_name in models_list:
        model = models[this_model_name]['estimator']
        explained_model = explain_model(model, predictors, target, folds, seed)
        explanations.loc[:, this_model_name] = explained_model.values

    return explanations
