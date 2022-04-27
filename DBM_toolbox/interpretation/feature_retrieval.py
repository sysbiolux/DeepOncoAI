import eli5
import pandas as pd
import logging
from DBM_toolbox.data_manipulation import dataset_class





def explain_model(model, dataset):
    pass





def explain_prediction(model, dataset, test_sample):
    pass




def explain_all(models, dataset):
    explanation_dict = {}

    for model in models:
        explained_model = explain_model(model, dataset)
        explanation_dict["models"][model] = explained_model
    dataframe = dataset.dataframe
    samples_list = dataframe.index
    for sample in samples_list:
        explained_prediction = explain_prediction(model, dataset, sample)
        explanation_dict["predictions"][sample] = explained_prediction

    return explanation_dict