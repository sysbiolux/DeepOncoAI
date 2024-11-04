
import numpy as np
import pandas as pd
from config import Config
from matplotlib import pyplot as plt
from DBM_toolbox.data_manipulation.data_utils import pickle_objects, unpickle_objects, merge_and_clean
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import glob
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


predictors = {
    'PD-0325901': ['ETV4', 'SPRY2', 'RP11-93B14.5', 'ETV5', 'ZNF502', 'SPRY1', 'TOR4A', 'CMTM7', 'DUSP6'],
    'PD0325901': ['ETV4', 'SPRY2', 'RP11-93B14.5', 'ETV5', 'ZNF502', 'SPRY1', 'TOR4A', 'CMTM7', 'DUSP6'],
    'Irinotecan': ['SLFN11', 'RP11-177C12.1', 'HNRNPA1', 'KHDC1'],
    'Erlotinib': ['RP11-902B17.1', 'RP11-47I22.1', 'IFI27', 'CORO2A', 'TSTD1'],
    'Lapatinib': ['RP11-902B17.1', 'RP11-47I22.1', 'PRKCH', 'ARHGAP27', 'DYRK3', 'SYTL1', 'GPX3', 'ADORA1', 'GPR135'],
}



### CCLE data:
old_dataset = unpickle_objects('FINAL_preprocessed_data_2023-02-16-10-30-39-935233.pkl')
config = Config("testall/config_explain.yaml")

final_data = unpickle_objects('FINAL_preprocessed_data_2023-02-16-10-30-39-935233.pkl')

missing_data = final_data.dataframe.loc[:, final_data.dataframe.isnull().any(axis=0)]

final_data = unpickle_objects("FINAL_explain_preprocessed_data_2024-06-25-21-07-10-041579.pkl")
trained_models = unpickle_objects("FINAL_explain_pre-models_2024-06-25-22-59-37-868197.pkl")

# training_labels = old_dataset.dataframe.index
#
# new_labels = final_data.dataframe.index
#
# validation_labels = [x for x in new_labels if x not in training_labels]

train_dataset = final_data

### GDSC:
df_drugs = pd.read_excel('data/Drugs_AUC_4drugs.xlsx')
df_pivoted = df_drugs.pivot(index=['Cell Line Name', 'Cosmic ID'], columns='Drug Name', values='AUC')
df_pivoted.columns = [f"{drug}_ActArea" for drug in df_pivoted.columns]
df_pivoted.reset_index(inplace=True)
print(df_pivoted.head())
df_pivoted.to_excel('Drugs_AUC_reformatted.xlsx', index=False)

df_genes = pd.read_csv('data/Cell_line_RMA_proc_basalExp_transposed.tsv', sep='\t')
df_genes = df_genes.iloc[1:].reset_index(drop=True)
df_genes.columns = ['Cosmic ID'] + list(df_genes.columns[1:])

df_genes.iloc[:, 1:] = df_genes.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

co = df_genes.columns[1:]

df_genes[co] = (df_genes[co] - df_genes[co].min()) / (df_genes[co].max() - df_genes[co].min())



df_merged = pd.merge(df_genes, df_pivoted, how='inner', on='Cosmic ID')
df_merged_clean = df_merged.loc[:, ~df_merged.columns.str.startswith('Unnamed')]
test_dataset = df_merged_clean
actarea_columns = df_merged_clean.filter(like='ActArea')

def quantize_column(col):
    return pd.qcut(col, q=3, labels=[0, 0.5, 1])

df_merged_clean[actarea_columns.columns] = actarea_columns.apply(quantize_column)

###########################

targets_list = ['PD-0325901_ActArea', 'Irinotecan_ActArea', 'Erlotinib_ActArea', 'Lapatinib_ActArea']

predictions = {}
conf_matrices = {}
models = {}
accuracies = pd.DataFrame(index=targets_list, columns=[''])

for target_name in targets_list:
    print(f"Target: {target_name}")
    this_predictors = predictors[target_name.split('_Act')[0]]
    colnames_predictors_train = [x for y in this_predictors for x in train_dataset.dataframe.columns if x.startswith(y+'_')]
    y_train = train_dataset.dataframe[target_name]
    X_train = train_dataset.dataframe.loc[:, colnames_predictors_train]
    X_train.columns = [x.split('_')[0] for x in X_train.columns]
    colnames_predictors_test = [x for y in this_predictors for x in test_dataset.columns if x.startswith(y)]
    if target_name == 'PD-0325901_ActArea':
        target_name = 'PD0325901_ActArea'
    y_test = test_dataset[target_name]
    X_test = test_dataset.loc[:, colnames_predictors_test]
    y_train_clean = y_train[(y_train == 0) | (y_train == 1)]
    X_train_clean = X_train.loc[y_train_clean.index]
    y_test_clean = y_test[(y_test == 0) | (y_test == 1)]
    X_test_clean = X_test.loc[y_test_clean.index]

    common_columns = X_train_clean.columns.intersection(X_test_clean.columns)
    X_train_clean = X_train_clean[common_columns]
    X_test_clean = X_test_clean[common_columns]

    print(f"train: {X_train_clean.shape[0]}")
    print(f"test: {X_test_clean.shape[0]}")

    # Initialize the classifiers
    linear_classifier = LinearRegression()
    logistic_classifier = LogisticRegression(random_state=42)
    decision_tree = DecisionTreeClassifier(random_state=42)
    rule_based_classifier = DummyClassifier(strategy="most_frequent", random_state=42)

    # Train the classifiers
    linear_classifier.fit(X_train_clean, y_train_clean)
    logistic_classifier.fit(X_train_clean, y_train_clean)
    decision_tree.fit(X_train_clean, y_train_clean)
    rule_based_classifier.fit(X_train_clean, y_train_clean)

    # Test the classifiers and get accuracy
    linear_accuracy = balanced_accuracy_score(y_test_clean, linear_classifier.predict(X_test_clean)>0.5)
    logistic_accuracy = balanced_accuracy_score(y_test_clean, logistic_classifier.predict(X_test_clean))
    decision_tree_accuracy = balanced_accuracy_score(y_test_clean, decision_tree.predict(X_test_clean))
    rule_based_accuracy = balanced_accuracy_score(y_test_clean, rule_based_classifier.predict(X_test_clean))

    # Print classifier performance
    print("Linear classifier accuracy:", linear_accuracy)
    print("Logistic classifier accuracy:", logistic_accuracy)
    print("Decision tree accuracy:", decision_tree_accuracy)
    print("Rule-based classifier accuracy:", rule_based_accuracy)
    classifier_names = ["linear", "logistic", "decision_tree", "dummy"]
    accuracies = [linear_accuracy, logistic_accuracy, decision_tree_accuracy, rule_based_accuracy]
    best_classifier_name = classifier_names[np.argmax(accuracies)]
    clf = {'linear': LinearRegression(),
           'logistic': LogisticRegression(random_state=42),
           'decision_tree': DecisionTreeClassifier(random_state=42),
           'dummy': DummyClassifier(strategy="most_frequent", random_state=42),
           }[best_classifier_name]
    print(f"Best classifier: {best_classifier_name}, with balanced accuracy {accuracies[np.argmax(accuracies)]}")
    clf.fit(X_train_clean, y_train_clean)

########### ############









    try:
        y_pred = clf.predict_proba(X_test_clean)[:, 1]
    except:
        y_pred = clf.predict(X_test_clean)
    predictions[target_name + '_truth'] = y_test_clean
    predictions[target_name] = y_pred
    fpr, tpr, thresholds = roc_curve(y_test_clean, y_pred)
    distances = np.sqrt(np.square(1 - tpr) + np.square(fpr))
    min_distance_index = np.argmin(distances)
    best_threshold = thresholds[min_distance_index]
    plt.subplots(figsize=(8, 8))
    hfont = {'fontname': 'Times New Roman'}
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=3, label=f"AUC: {round(roc_auc, 3)}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.plot(fpr[min_distance_index], tpr[min_distance_index], marker='o', color='blue', label='Best Threshold = %0.2f' % best_threshold, markersize=6)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=25, **hfont)
    plt.ylabel("True Positive Rate", fontsize=25, **hfont)
    plt.title(f"expl_{best_classifier_name}_{target_name.split('_')[0]}_n={len(y_pred)}", fontsize=25, **hfont)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(f'FINAL_redo_Expl_ROC_{target_name}.tif')
    plt.close()
    # plt.subplots(figsize=(20, 20))
    # feat_names = [x.split('_')[0] for x in clf.feature_names_in_]
    # tree.plot_tree(clf, feature_names=feat_names)
    # plt.savefig(f'FINAL_ExplDT_struct_{target_name}.tif')
    # plt.close()
    #
    # predictions['model'] = clf.copy()
    #
    # accuracy = accuracy_score(y_test_clean, y_pred)
    # ba = balanced_accuracy_score(y_test_clean, y_pred)
    # report = classification_report(y_test_clean, y_pred)
    y_pred = y_pred > best_threshold
    conf_matrix = confusion_matrix(y_test_clean, y_pred)
    conf_matrices[target_name] = conf_matrix
    models[target_name] = clone(clf)
    feature_names_all = clf.feature_names_in_
    try:
        feature_score_all = np.abs(clf.coef_)
    except:
        feature_score_all = clf.feature_importances_
    if len(feature_score_all.shape) > 1:
        feature_score_all = feature_score_all.ravel()
    top3_indices = np.argsort(feature_score_all)[-3:]
    top3_features = [feature_names_all[i] for i in top3_indices]

    feature1_idx = top3_indices[2]
    feature2_idx = top3_indices[1]
    fixed_feature_idx = top3_indices[0]

    percentiles = [10, 25, 50, 75, 90]
    fixed_feature_values = np.percentile(X_train_clean.loc[:, top3_features[2]], percentiles)

    # Set up the figure and the subplots
    fig, axs = plt.subplots(1, len(percentiles), figsize=(18, 4))

    for i, fixed_value in enumerate(fixed_feature_values):
        feature1_values, feature2_values, pdp_results = partial_dependency(clf, X_train_clean, feature1_idx, feature2_idx,
                                                                           fixed_feature_idx, fixed_value)

        # Create a contour plot
        x, y = np.meshgrid(feature1_values, feature2_values)
        cs = axs[i].contourf(x, y, pdp_results, cmap=plt.cm.coolwarm)

        # Customize the subplot
        axs[i].set_title(f'{percentiles[i]}th percentile of {top3_features[0].split("_")[0]}')
        axs[i].set_xlabel(top3_features[2].split("_")[0])
        axs[i].set_ylabel(top3_features[1].split("_")[0])

    # Add a colorbar to the figure
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cs, cax=cbar_ax)
    plt.savefig(f'FINAL_redo_Expl_PDP_{target_name}.tif')
    plt.close()



