import eli5
import pandas as pd
import logging
from DBM_toolbox.data_manipulation import dataset_class


def explain_model(model, predictors, target):
    logging.info("explaining models")
    trained_model = model.fit(predictors, target)
    explanation = eli5.explain_weights_df(trained_model)
    return explanation


def explain_prediction(model, predictors, target, test_sample):
    logging.info("starting prediction explanations")
    trained_model = model.fit(predictors, target)
    explanation = eli5.explain_prediction(trained_model, test_sample)
    return explanation


def explain_all(models, predictors, target):
    explanation_dict = {}

    logging.info("starting model explanations...")
    for model in models:
        explained_model = explain_model(model, predictors, target)
        explanation_dict["models"][model] = explained_model
        samples_list = target.index
        for sample in samples_list:
            explained_prediction = explain_prediction(model, predictors, target, sample)
            explanation_dict["predictions"][sample][model] = explained_prediction

    return explanation_dict
