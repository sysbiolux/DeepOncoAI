# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:14:31 2020

@author: sebde
"""

from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from scripts.modeling.stacking import stack_models


def plot_roc(y_test, X_test, y_train, X_train, thisNames, models, item=0, ax=None):
    try:
        model = models[thisNames[item]]
        model.fit(X_train, y_train)
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict(X_test))
        if ax == None:
            fig, ax = plt.subplots()
        plt.tight_layout
        ax.plot(fpr, tpr, label="%s ROC (area = %0.2f)" % (thisNames[item], auc))
    except:
        print("The method %s does not support ROC curves" % thisNames[item])
    return ax


def plot_roc_validation(X, y, final_model, models, reps=5, folds=5, ax=None):
    """ performs (reps) times (folds)-fold cross-validation and displays 
	the average ROC curve, its confidence interval as well as each ROC curve
	 for each validation fold for an out-of-bag stacked model.
	Parameters
	----------
	X : DataFrame
		Predictors
	y : Series
		Targets
	final_model : model
		sklearn model type for stacking
	models : List
		list of sklearn models to be stacked
	reps : int, optional
		number of shuffled repeats. The default is 5.
	folds : int, optional
		number of folds in the cross-validation. The default is 5.
	ax : axis, optional
		the axis to be plotted. The default is None.

	Returns
	-------
	None.

	"""
    try:
        tprs = []
        scores = []
        base_fpr = np.linspace(0, 1, 101)
        AllCorrect = np.zeros((1, X.shape[0]))
        idx = np.arange(0, len(y))
        correct = np.zeros((X.shape[0]))
        np.random.seed(42)
        for j in np.random.randint(0, high=10000, size=reps):
            np.random.shuffle(idx)
            kf = StratifiedKFold(folds, shuffle=True, random_state=j)
            X_shuff = X.iloc[idx, :]
            y_shuff = y.iloc[idx]
            for train, test in kf.split(X_shuff, y_shuff):
                X_train = X_shuff.iloc[train]
                y_train = y_shuff.iloc[train]
                X_test = X_shuff.iloc[test]
                y_test = y_shuff.iloc[test]
                if models == []:
                    trained_model = final_model.fit(X_train, y_train)
                    y_score = trained_model.predict_proba(X_test)
                    y_pred = trained_model.predict(X_test)
                else:
                    stacked_model, perf, train_perf, y_pred, y_score = stack_models(
                        models, final_model, X_train, y_train, X_test, y_test, folds
                    )
                # 				y_score = stacked_model.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                score = roc_auc_score(y_test, y_score[:, 1])
                scores.append(score)
                ax.plot(fpr, tpr, "b", alpha=0.1)
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                tprs.append(tpr)
                ok = y_test == y_pred
                allz = np.zeros((X.shape[0]))
                allz[test] = ok
                correct = correct + allz
        AllCorrect[0, :] = correct / reps
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        mean_scores = np.mean(scores)
        std_scores = np.std(scores)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        ax.plot(base_fpr, mean_tprs, "b")
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3)
        ax.plot([0, 1], [0, 1], "r--")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])

        ax.set_title(
            "AUC in %d-times %d-fold CV: %.3f +- %.3f"
            % (reps, folds, mean_scores, std_scores)
        )
    except:
        print("problem here")

    return ax, mean_scores, std_scores
