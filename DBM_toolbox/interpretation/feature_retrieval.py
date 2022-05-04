from eli5.sklearn import PermutationImportance
import pandas as pd
import logging
from DBM_toolbox.data_manipulation import dataset_class


def explain_model(model, predictors, target, folds=5, seed=42):
    logging.info("explaining models")
    trained_model = model.fit(predictors, target)
    explanation = PermutationImportance(estimator=trained_model, random_state=seed, cv=folds)
    explanation.fit(predictors, target)
    return explanation


def explain_prediction(model, predictors, target, test_sample):
    logging.info("starting prediction explanations")
    trained_model = model.fit(predictors, target)
    explanation = eli5.explain_prediction(trained_model, test_sample)
    return explanation


def explain_all(models, predictors, target, folds=5, seed=42):
    explanations = {}

    logging.info("starting model explanations...")
    print("*******************************")
    print(models)
    models_list = models.keys()
    for this_model_name in models_list:
        model = models[this_model_name]['estimator']
        explained_model = explain_model(model, predictors, target, folds, seed)
        explanations["model_explained"][model] = explained_model
        samples_list = target.index
        for sample in samples_list:
            explained_prediction = explain_prediction(model, predictors, target, sample)
            explanations["predictions_explained"][sample][model] = explained_prediction

    return explanations
