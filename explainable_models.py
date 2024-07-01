
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D


def partial_dependency(model, X, feature1_idx, feature2_idx, fixed_feature_idx, fixed_feature_value,
                       grid_resolution=100):
    feature1_values = np.linspace(X.iloc[:, feature1_idx].min(), X.iloc[:, feature1_idx].max(), grid_resolution)
    feature2_values = np.linspace(X.iloc[:, feature2_idx].min(), X.iloc[:, feature2_idx].max(), grid_resolution)

    xx, yy = np.meshgrid(feature1_values, feature2_values)
    pdp_data = np.c_[xx.ravel(), yy.ravel()]

    X_temp = X.copy()
    pdp_results = np.zeros((grid_resolution, grid_resolution))

    X_temp.iloc[:, fixed_feature_idx] = fixed_feature_value

    for i, f1_val in enumerate(feature1_values):
        for j, f2_val in enumerate(feature2_values):
            X_temp.iloc[:, feature1_idx] = f1_val
            X_temp.iloc[:, feature2_idx] = f2_val
            pdp_results[j, i] = np.mean(model.predict(X_temp))

    return feature1_values, feature2_values, pdp_results


old_dataset = unpickle_objects('FINAL_preprocessed_data_2023-02-16-10-30-39-935233.pkl')
old_final_results = unpickle_objects('FINAL_results_2023-02-20-12-35-21-577662.pkl')
config = Config("testall/config_explain.yaml")


logging.info("Reading data")
data, ActAreas, ic50s, dose_responses = config.read_data()

logging.info("Filtering data")
filtered_data, filters = config.filter_data(data)

#####

logging.info("Selecting subsets for feature engineering")
selected_subset = config.select_subsets(filtered_data)

logging.info("Engineering features")
engineered_features = config.engineer_features(filtered_data)

logging.info("Merging engineered features")
engineered_data = filtered_data.merge_with(engineered_features)

logging.info("Quantizing targets")
quantized_data = config.quantize(engineered_data, target_omic="DRUGS", ic50s=ic50s)

final_data = quantized_data.normalize().optimize_formats()
config.save(to_save=final_data, name="FINAL_explain_preprocessed_data")

missing_data = final_data.dataframe.loc[:, final_data.dataframe.isnull().any(axis=0)]

######

logging.info("Getting optimized models")

trained_models = config.get_models(dataset=final_data, method="standard")
config.save(to_save=trained_models, name="FINAL_explain_pre-models")

###########
final_data = unpickle_objects("FINAL_explain_preprocessed_data_2024-06-25-21-07-10-041579.pkl")
trained_models = unpickle_objects("FINAL_explain_pre-models_2024-06-25-22-59-37-868197.pkl")


predictors = {
    'PD-0325901': ['ETV4', 'SPRY2', 'RP11-93B14.5', 'ETV5', 'ZNF502', 'SPRY1', 'TOR4A', 'CMTM7', 'DUSP6'],
    'AZD6244': ['ETV4', 'SPRY2', 'TMC6', 'RP11-93B14.5', 'RP11-712B9.2', 'RP11-973F15.1', 'CMTM7', 'S100A4', 'TRPV2', 'SPRY4'],
    'Paclitaxel': ['RPUSD2', 'MAGEA6', 'SLFN11', 'GPX2', 'DUT', 'LEPREL2', 'RCOR2', 'DAAM1'],
    'Panobinostat': ['TNFRSF12A', 'AJUBA', 'IFITM3', 'RP11-783K16.5', 'RP11-7F17.7', 'AC138623.1', 'AC011242.6', 'ZNF215'],
    'Irinotecan': ['SLFN11', 'RP11-177C12.1', 'HNRNPA1', 'KHDC1'],
    'Erlotinib': ['RP11-902B17.1', 'RP11-47I22.1', 'IFI27', 'CORO2A', 'TSTD1'],
    'Lapatinib': ['RP11-902B17.1', 'RP11-47I22.1', 'PRKCH', 'ARHGAP27', 'DYRK3', 'SYTL1', 'GPX3', 'ADORA1', 'GPR135'],
    'TAE684': ['C14orf37', 'TUBGCP3', 'DUT', 'BLM', 'PXDN', 'GLB1L2', 'PXDN', 'PPARG', 'ARID3A', 'TNFRSF1B'],
    'PD-0332991': ['RP11-7F17.7', 'DUT', 'PDLIM4', 'NDN', 'ACOT1', 'ACOT2', 'GPX2'],
    '17-AAG': ['NQO1', 'MB21D1', 'SLC16A3'],
    'RAF265': ['IFI27', 'FOXA1', 'DNASE1L2', 'LEPREL2', 'CHRNB1', 'HERC5', 'MYEOV', 'PLEKHG4', 'ABLIM2'],
    'TKI258': ['RP11-950C14.3', 'SERPINA5', 'LEPREL2', 'TSPYL5', 'EMR1'],
    'Nilotinib': ['GPX2', 'HIF1A', 'RP11-7F17.7', 'ITGA9', 'GGT6', 'AGAP2', 'RP11-47I22.2'],
    'PF2341066': ['PTGER2', 'CRIP1', 'GPX2', 'ANP32A', 'NDN', 'HGF', 'S100A4', 'DACT1'],
    'ZD-6474': ['TGFB3', 'FKBP3', 'PCDHB2', 'PCDHB6', 'MCALL1', 'Xxbac-BPG181B23.7'],
    'AEW541': ['IGF1R', 'MAGEA1', 'GPX3', 'SNAPC1'],
    'PLX4720': ['CKB', 'SETD3', 'APOPT1', 'AKAP5', 'XAGE1A', 'GPX3', 'CCND2', 'XAGE1E', 'DAAM1', 'RSPH1', 'GPR135'],
    'L-685458': ['RP11-85G20.1', 'CCNB1IP1', 'ZNF625', 'ZSCAN18'],
    'Sorafenib': ['RPL4', 'TM9SF1', 'FAM21B', 'PYCARD', 'HGMA2', 'PTEN', 'NTN4'],
    'AZD0530': ['IFI27', 'PREX2', 'PLAGL1', 'GSTT1'],
    'PHA-665752': ['TDRD9', 'PTGER2', 'HN1L', 'SLC6A15', 'RP11-408B11.2', 'CORO2A', 'CELF2-AS1', 'ARID4A', 'HNRNPCP1'],
    'Nutlin-3': ['DDB2', 'NMD2', 'PTPN13', 'EDA2R'],
    'LBW242': ['FBLN2', 'RP11-8L2.1', 'FBLN2', 'HECA'],
}

training_labels = old_dataset.dataframe.index

new_labels = final_data.dataframe.index

validation_labels = [x for x in new_labels if x not in training_labels]

train_dataset, test_dataset = final_data.split(train_index=training_labels, test_index=validation_labels)

targets_list = final_data.to_pandas(omic='DRUGS').columns

predictions = {}
conf_matrices = {}
models = {}
accuracies = pd.DataFrame(index=targets_list, columns=[''])

for target_name in targets_list:
    print(f"Target: {target_name}")
    this_predictors = predictors[target_name.split('_Act')[0]]
    colnames_predictors = [x for y in this_predictors for x in train_dataset.dataframe.columns if x.startswith(y+'_')]
    y_train = train_dataset.dataframe[target_name]
    X_train = train_dataset.dataframe.loc[:, colnames_predictors]
    y_test = test_dataset.dataframe[target_name]
    X_test = test_dataset.dataframe.loc[:, colnames_predictors]
    y_train_clean = y_train[(y_train == 0) | (y_train == 1)]
    X_train_clean = X_train.loc[y_train_clean.index]
    y_test_clean = y_test[(y_test == 0) | (y_test == 1)]
    X_test_clean = X_test.loc[y_test_clean.index]

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



