# -*- coding: utf-8 -*-
### THIS SCRIPT IS DEPRECATED ###
"""
Created on Sat Nov 21 11:20:01 2020

@author: sebde
"""
import numpy as np
import time
import warnings
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import rotation_forest as rot
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import cross_val_score


def get_classification_models(models=dict(), depth=1):
    # linear models
    #     models['linear'] = LinearRegression()
    models["logistic"] = LogisticRegression()
    if depth == 1:
        alpha = [0.01, 0.1, 0.5, 1.0, 2.0, 10, 20, 100]
    else:
        alpha = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10, 20, 100]

    for a in alpha:
        models["ridge-" + str(a)] = RidgeClassifier(alpha=a)

    models["sgd"] = SGDClassifier(max_iter=1000, tol=1e-3)
    models["pa"] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
    # non-linear models
    if depth == 1:
        n_neighbors = [1, 3, 5, 7, 11, 21, 31]
    else:
        n_neighbors = range(1, 100)

    for k in n_neighbors:
        models["knn-" + str(k)] = KNeighborsClassifier(n_neighbors=k)

    models["near"] = NearestCentroid()
    models["cart"] = DecisionTreeClassifier()
    models["extra"] = ExtraTreeClassifier()
    models["rbf"] = GaussianProcessClassifier(1.0 * RBF(1.0))
    models["qda"] = QuadraticDiscriminantAnalysis()
    models["svml"] = SVC(kernel="linear", probability=True)
    models["svmp"] = SVC(kernel="poly", probability=True)
    if depth == 1:
        c_values = [0.001, 0.2, 0.5, 0.8, 0.99]
    else:
        c_values = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]

    for c in c_values:
        models["svmr" + str(c)] = SVC(C=c, probability=True, gamma="auto")
    models["bayes"] = GaussianNB()
    # ensemble models
    if depth == 1:
        n_trees = 200
    else:
        n_trees = 1000
    models["ada"] = AdaBoostClassifier(n_estimators=n_trees)
    models["bag"] = BaggingClassifier(n_estimators=n_trees)
    models["rf"] = RandomForestClassifier(n_estimators=n_trees)
    models["et"] = ExtraTreesClassifier(n_estimators=n_trees)
    models["gbm"] = GradientBoostingClassifier(n_estimators=n_trees)
    models["xgb"] = xgb.XGBClassifier(max_depth=3, n_estimators=n_trees)
    models["xgbmax1"] = xgb.XGBClassifier(max_depth=4, n_estimators=n_trees)
    models["xgbslo1"] = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.02, n_estimators=n_trees
    )
    if depth > 2:
        n_trees = n_trees * 2

    if depth > 1:
        models["rot"] = rot.RotationForestClassifier(n_estimators=n_trees)
        models["rotrnd"] = rot.RotationForestClassifier(
            n_estimators=n_trees, rotation_algo="randomized"
        )
        models["adamax1"] = AdaBoostClassifier(n_estimators=n_trees, learning_rate=0.1)
        models["rfmax1"] = RandomForestClassifier(
            n_estimators=n_trees, criterion="entropy", min_samples_split=4
        )
        models["etmax1"] = ExtraTreesClassifier(
            n_estimators=n_trees, bootstrap=True, criterion="entropy"
        )
        models["gbmmax1"] = GradientBoostingClassifier(
            n_estimators=n_trees, learning_rate=0.02
        )
        models["xgbmax2"] = xgb.XGBClassifier(
            max_depth=5, n_estimators=n_trees, nthread=-1
        )

    if depth > 1:
        n1_values = [2, 4, 8]
        n2_values = [4, 8, 16]
        n3_values = [2, 4, 8]
        nMax = 1000
        if depth >= 2:
            n1_values = [2, 4, 8, 16, 32]
            n2_values = [4, 8, 16, 32, 64]
            n3_values = [2, 4, 8, 16, 32]
            nMax = 10000
        for n1 in n1_values:
            for n2 in n2_values:
                for n3 in n3_values:
                    models[
                        "mlp" + str(n1) + "-" + str(n2) + "-" + str(n3)
                    ] = MLPClassifier(
                        solver="sgd",
                        learning_rate="adaptive",
                        learning_rate_init=0.01,
                        hidden_layer_sizes=(n1, n2, n3),
                        verbose=True,
                        tol=0.00001,
                        n_iter_no_change=nMax / 2,
                        batch_size=32,
                        max_iter=nMax,
                    )

    print("Defined %d models" % len(models))
    return models


def get_regression_models(models=dict(), depth=1):
    # linear models
    models["lr"] = LinearRegression()
    if depth < 2:
        alpha = [0.0, 0.01, 0.5, 2]
    else:
        alpha = [0.0, 0.01, 0.1, 0.2, 0.5, 0.7, 1, 2]

    if depth < 2:
        l1ratio = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
    else:
        l1ratio = [
            0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.33,
            0.5,
            0.66,
            0.8,
            0.9,
            0.95,
            0.98,
            0.99,
            1,
        ]
    for a in alpha:
        for b in l1ratio:
            models["enet" + str(a) + "-" + str(b)] = ElasticNet(alpha=a, l1_ratio=b)
    for a in alpha:
        models["lasso-" + str(a)] = Lasso(alpha=a)
    for a in alpha:
        models["ridge-" + str(a)] = Ridge(alpha=a)

    if depth > 1:
        models["huber"] = HuberRegressor()
        models["lars"] = Lars()
        models["llars"] = LassoLars()
        models["pa"] = PassiveAggressiveRegressor(max_iter=10000, tol=1e-4)
        models["ranscac"] = RANSACRegressor()
        models["sgd"] = SGDRegressor(max_iter=10000, tol=1e-4)
        models["theil"] = TheilSenRegressor(n_jobs=-1)

    # non-linear models
    if depth < 2:
        n_neighbors = [2, 5, 7, 20]  # check if it makes sense to vary this more...
    else:
        n_neighbors = [2, 3, 4, 5, 7, 10, 20, 50]  # check if this is enough...

    for k in n_neighbors:
        models["knn-" + str(k)] = KNeighborsRegressor(n_neighbors=k)
    models["cart"] = DecisionTreeRegressor()
    models["extra"] = ExtraTreeRegressor()
    models["svml"] = SVR(kernel="linear")
    models["svmp"] = SVR(kernel="poly")
    if depth == 1:
        c_values = [0.01, 0.2, 0.5, 0.75, 0.99, 1]
    else:
        c_values = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]

    for c in c_values:
        models["svmr" + str(c)] = SVR(C=c)
    # ensemble models
    if depth < 2:
        n_trees = 100
    else:
        n_trees = 1000

    models["ada"] = AdaBoostRegressor(n_estimators=n_trees)
    models["bag"] = BaggingRegressor(n_estimators=n_trees, n_jobs=-1)
    models["rf"] = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
    models["et"] = ExtraTreesRegressor(n_estimators=n_trees, n_jobs=-1)
    models["gbm"] = GradientBoostingRegressor(n_estimators=n_trees)
    models["xgb"] = xgb.XGBRegressor(n_estimators=n_trees, nthread=-1)
    if depth > 1:
        n1_values = [2, 4, 8]
        n2_values = [4, 8, 16]
        n3_values = [2, 4, 8]
        nMax = 1000
        if depth >= 2:
            n1_values = [2, 4, 8, 16, 32]
            n2_values = [4, 8, 16, 32, 64]
            n3_values = [2, 4, 8, 16, 32]
            nMax = 10000
        for n1 in n1_values:
            for n2 in n2_values:
                for n3 in n3_values:
                    models[
                        "mlp" + str(n1) + "-" + str(n2) + "-" + str(n3)
                    ] = MLPRegressor(
                        solver="sgd",
                        learning_rate="adaptive",
                        learning_rate_init=0.01,
                        hidden_layer_sizes=(n1, n2, n3),
                        verbose=False,
                        tol=0.00001,
                        n_iter_no_change=nMax / 2,
                        batch_size=32,
                        max_iter=nMax,
                    )
    print("Defined %d models" % len(models))
    return models


# create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = list()
    # standardization
    #    steps.append(('standardize', StandardScaler()))
    # normalization
    #    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(("model", model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


def evaluate_model(X, y, model, folds, metric):
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds)
    return scores


def robust_evaluate_model(X, y, model, X_test, folds, metric):
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, folds, metric)
            fitModel = model.fit(X, y)
            predictions = fitModel.predict(X_test)
    except:
        scores = None
        predictions = None
    return (scores, predictions)


def evaluate_models(X, y, models, X_test, folds=10, metric="accuracy"):
    """Evaluate a list of models with x-fold CV"""
    results = dict()
    predicted = dict()
    for name, model in models.items():
        time_start = time.clock()
        scores, predictions = robust_evaluate_model(X, y, model, X_test, folds, metric)
        timesec = time.clock() - time_start
        if scores is not None:
            results[name] = scores
            predictions[predictions > 0.5] = 1
            predictions[predictions <= 0.5] = 0
            predicted[name] = predictions
            mean_score, std_score, = np.mean(scores), np.std(scores)
            print(
                ">%s: %0.0f-val= %.3f +/- %.3f in %.3fs"
                % (name, folds, mean_score, std_score, timesec)
            )
        #             logging.debug('>%s: %.3f (+/- %.3f) in %.3fs' % (name, mean_score, std_score, timesec))
        else:
            print(">%s: error" % name)
    return (results, predicted)


def summarize_results(
    results, predicted, y_test, thisCol, maximize=True, top_n=0, graph=False
):
    """summarizes the results for the top-performing algorithms"""
    if top_n == 0:
        top_n = len(results)
    if len(results) == 0:
        print("no results")
        return
    n = min(top_n, len(results))
    mean_scores = [(k, np.mean(v)) for k, v in results.items()]
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    if maximize:
        mean_scores = list(reversed(mean_scores))
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    predict = [predicted[x[0]] for x in mean_scores[:n]]
    #     logging.debug('Target = %s' %(thisCol))
    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = np.mean(results[name]), np.std(results[name])
    #         logging.debug('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
    #         logging.debug(confusion_matrix(y_test, predicted[name]))
    if graph:
        f, axes = plt.subplots(1, 1, figsize=(30, 50))
        plt.boxplot(scores, labels=names)
        _, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        thisTitle = thisCol + "_spotcheck.pdf"
        plt.savefig(thisTitle)
    return names, scores, predict
