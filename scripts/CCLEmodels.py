# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""


# Importing the libraries
import warnings
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.neural_network import MLPRegressor
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import scipy as sc
from scipy.stats import norm
import numba
import platform
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

sns.set(context='talk')
print("Operating system:", platform.system(), platform.release())
import sys
print("Python version:", sys.version)
print("Numpy version:", np.version.version)
print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Numba version:", numba.__version__)


#Reduce dataframe memory usage
def reduce_mem_usage(df):
	""" iterate through all the columns of a dataframe and modify the data type
		to reduce memory usage.
	"""
	start_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

	for col in df.columns:
		col_type = df[col].dtype

		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('category')

	end_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

	return df


#Load data
dfProt = pd.read_csv('CCLE_RPPA_20181003.csv')
dfMeta = pd.read_csv('CCLE_metabolomics_20190502.csv')
dfDrug = pd.read_csv('CCLE_NP24.2009_Drug_data_2015.02.24.csv')

dfProt_dropped = dfProt.drop(columns=['Unnamed: 0'])
dfMeta_dropped = dfMeta.drop(columns=['CCLE_ID', 'DepMap_ID'])
dfDrug_dropped = dfDrug.drop(columns=['CCLE Cell Line Name', 'Primary Cell Line Name', 'Compound', 'Target', 'Doses (uM)', 'Activity Data (median)', 'Activity SD', 'Num Data', 'FitType'])


#Sometimes it changes some values in the dataframe, let's check it doesnt' change anything
df_test = pd.DataFrame()
dfProt_opt = reduce_mem_usage(dfProt)

for col in dfProt_dropped:
    df_test[col] = dfProt[col] - dfProt_opt[col]

#Mean, max and min for all columns should be 0
df_test.describe().loc['mean']
df_test.describe().loc['max']
df_test.describe().loc['min']

df_test = pd.DataFrame()
dfMeta_opt = reduce_mem_usage(dfMeta)

for col in dfMeta_dropped:
    df_test[col] = dfMeta[col] - dfMeta_opt[col]

#Mean, max and min for all columns should be 0
df_test.describe().loc['mean']
df_test.describe().loc['max']
df_test.describe().loc['min']

df_test = pd.DataFrame()
dfDrug_opt = reduce_mem_usage(dfDrug)

for col in dfDrug_dropped:
    df_test[col] = dfDrug[col] - dfDrug_opt[col]

#Mean, max and min for all columns should be 0
df_test.describe().loc['mean']
df_test.describe().loc['max']
df_test.describe().loc['min']

#Acceleration
@numba.jit
def f(x):
	return x
@numba.njit
def f(x):
	return x


#Extract drug-specific data
drugNames = dfDrug_opt['Compound'].unique()
dfDrug_opt['Compound'].value_counts()

dfProt_opt = dfProt_opt.rename(columns={"Unnamed: 0": "CCLE Cell Line Name"})

#Look for missing prot data in rows
total_null = dfProt_opt.isna().sum(axis = 1).sort_values(ascending=False)
percent = 100*(dfProt_opt.isna().sum(axis = 1)/dfProt_opt.isna().count(axis = 1)).sort_values(ascending=False)
missing_protrow_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
#All rows are complete

#Look for missing drug data in rows
total_null = dfDrug_opt.isna().sum(axis = 1).sort_values(ascending=False)
percent = 100*(dfDrug_opt.isna().sum(axis = 1)/dfDrug_opt.isna().count(axis = 1)).sort_values(ascending=False)
missing_drugrow_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])

#Look for missing drug data in columns
total_null = dfDrug_opt.isna().sum().sort_values(ascending=False)
percent = 100*(dfDrug_opt.isna().sum()/dfDrug_opt.isna().count()).sort_values(ascending=False)
missing_drugcolumn_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
#38% missing data in EC50: drop this column
dfDrug_opt = dfDrug_opt.drop(columns=['EC50 (uM)'])


Merged = dfProt_opt

for thisDrug in drugNames:
	dfDrug_opt_spec = dfDrug_opt.loc[dfDrug_opt['Compound'] == thisDrug]
	dfDrug_opt_spec_clean = dfDrug_opt_spec.drop(columns =['Primary Cell Line Name', 'Compound', 'Target', 'Doses (uM)', 'Activity Data (median)', 'Activity SD', 'Num Data', 'FitType'])
	dfDrug_opt_spec_clean.columns=['CCLE Cell Line Name', thisDrug+'_IC50', thisDrug+'_Amax', thisDrug+'_ActArea']

	#Merge dataset
	Merged = pd.merge(Merged, dfDrug_opt_spec_clean, how='left', on='CCLE Cell Line Name', sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)

df = Merged.set_index('CCLE Cell Line Name')

#Check for missing data in rows
total_null = df.isna().sum(axis = 1).sort_values(ascending=False)
percent = 100*(df.isna().sum(axis = 1)/df.isna().count(axis = 1)).sort_values(ascending=False)
missing_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
#Drop rows for which no drug data is available
df = df.dropna(thresh=286-71)
#441 rows deleted (49%)

#Check for missing data in columns
total_null = df.isna().sum().sort_values(ascending=False)
percent = 100*(df.isna().sum()/df.isna().count()).sort_values(ascending=False)
missing_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
#Irinotecan might pose a problem later on...

#Exploratory Data Analysis (EDA)
#1) Inputs
corr = df.iloc[:,1:213].corr()
np.fill_diagonal(corr.values,0)
f, ax = plt.subplots(2, 1, figsize=(15, 12))
sns.heatmap(corr, ax=ax[0], center=0)
plt.title('Inputs Correlations')
sns.distplot(corr.values.ravel(), ax = ax[1])
plt.title('Distribution of correlations')
plt.savefig('Inputs_Correlations.png')
#There are very few correlated variables
#2) Outputs
corr = df.iloc[:,214:].corr()
np.fill_diagonal(corr.values,0)
f, ax = plt.subplots(2, 1, figsize=(15, 12))
sns.heatmap(corr, ax=ax[0], center=0)
plt.title('Outputs Correlations')
#Seems like Act area give the inverse info as IC50 and AMax, which are highly correlated (except for Panobinostat, strangely)
sns.distplot(corr.values.ravel(), ax=ax[1])
plt.title('Distribution of correlations')
plt.savefig('Outputs_Correlations.png')
#We will predict ActArea, let's drop the other columns
colsToDelete = [214, 215, 217, 218, 220, 221, 223, 224, 226, 227, 229, 230, 232, 233, 235, 236, 238, 239, 241, 242, 244, 245, 247, 248, 250, 251, 253, 254, 256, 257, 259, 260, 262, 263, 265, 266, 268, 269, 271, 272, 274, 275, 277, 278, 280, 281, 283, 284]
df_reduced = df.drop(df.columns[colsToDelete],axis=1)
#Investigate Outputs
corr = df_reduced.iloc[:,214:-1].corr()
np.fill_diagonal(corr.values,0)
f, ax = plt.subplots(2, 1, figsize=(15, 12))
sns.heatmap(corr, ax=ax[0], center=0)
plt.title('Outputs Correlations')
sns.distplot(corr.values.ravel(), ax=ax[1])
plt.title('Distribution of correlations')
plt.savefig('Outputs_Correlation2.png')

#Irinotecan and topotecan are related (TopoI inhibitors)
#AZD6244 and PD0325901 are related (MEK inhibitors)

plotOutputs = sns.pairplot(data=df_reduced.iloc[:,214:-1], diag_kind="kde", markers="+")
plt.savefig('Outputs_Plots2.png')

#Normality check
colnames = list(df_reduced.iloc[:,214:].columns)

f, axes = plt.subplots(4, 6, figsize=(15, 12))

for i in range(4):
	for j in range(6):
		col = i*6+j
		thisdata = df_reduced[colnames[col]]
		df_reduced[colnames[col]+'_log'] = np.log(df_reduced[colnames[col]]-min(thisdata)+1)
		thisdata = thisdata.dropna()
		sns.distplot(thisdata, fit=norm, label='original', ax=axes[i,j]);
		sns.distplot(np.log(thisdata-min(thisdata)+0.01), fit=norm, label='log', ax=axes[i,j])
		f.legend()

plt.savefig('Inputs_dist.png')
#Not sur which one is best: keeping the log-transformed outputs as alternatives
#Investigate log Outputs
corr = df_reduced.iloc[:,238:-1].corr()
np.fill_diagonal(corr.values,0)
f, ax = plt.subplots(2, 1, figsize=(15, 12))
sns.heatmap(corr, ax=ax[0], center=0)
plt.savefig('Outputswithlogs_Correlation.png')
plotOutputs = sns.pairplot(data=df_reduced.iloc[:,238:-1], diag_kind="kde", markers="+")
plt.savefig('Outputswithlogs_Pair.png')


#Check outliers
f, ax = plt.subplots(figsize=(15, 12))
df_reduced.boxplot(column=list(df_reduced.iloc[:,214:237].columns))
plt.savefig('Outputs_boxplots.png')
f, ax = plt.subplots(figsize=(15, 12))
df_reduced.boxplot(column=list(df_reduced.iloc[:,238:-1].columns))
plt.savefig('Outputslogs_boxplots.png')
#No worrying outlier

#Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_reduced)
df_norm = pd.DataFrame(scaler.transform(df_reduced), columns=df_reduced.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_norm)
df_norm_scaled = pd.DataFrame(scaler.transform(df_norm), columns=df_norm.columns)

#Check range
f, ax = plt.subplots(figsize=(15, 12))
df_norm_scaled.boxplot(column=list(df_norm_scaled.iloc[:,214:237].columns))
plt.savefig('Outputs_normscaled_boxplots.png')
f, ax = plt.subplots(figsize=(15, 12))
df_norm_scaled.boxplot(column=list(df_norm_scaled.iloc[:,238:-1].columns))
plt.savefig('Outputslogs_normscaled_boxplots.png')
################

##########################################################################
# regression spot check script

# create a dict of standard models to evaluate {name:object}
def get_models(models=dict()):
	# linear models
	models['lr'] = LinearRegression()
	alpha = [0.0, 0.01, 0.1, 0.2, 0.5, 0.7, 1, 2]
	for a in alpha:
		models['lasso-'+str(a)] = Lasso(alpha=a)
	for a in alpha:
		models['ridge-'+str(a)] = Ridge(alpha=a)
	for a1 in alpha:
		for a2 in alpha:
			name = 'en-' + str(a1) + '-' + str(a2)
			models[name] = ElasticNet(a1, a2)
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=10000, tol=1e-4)
	models['ranscac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=10000, tol=1e-4)
	models['theil'] = TheilSenRegressor(n_jobs=-1)
	# non-linear models
	n_neighbors = [1, 2, 3, 5, 7, 10, 20]
	for k in n_neighbors:
		models['knn-'+str(k)] = KNeighborsRegressor(n_neighbors=k)
	models['cart'] = DecisionTreeRegressor()
	models['extra'] = ExtraTreeRegressor()
	models['svml'] = SVR(kernel='linear')
	models['svmp'] = SVR(kernel='poly')
	c_values = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
	for c in c_values:
		models['svmr'+str(c)] = SVR(C=c)
	# ensemble models
	n_trees = 500
	models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
	models['bag'] = BaggingRegressor(n_estimators=n_trees, n_jobs=-1)
	models['rf'] = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
	models['et'] = ExtraTreesRegressor(n_estimators=n_trees, n_jobs=-1)
	models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
	models['xgb'] = xgb.XGBRegressor(n_estimators=n_trees, nthread=-1)
	n1_values = [1, 10, 100]
	n2_values = [1, 10, 100]
	n3_values = [1, 10, 100]
	for n1 in n1_values:
		for n2 in n2_values:
			for n3 in n3_values:
				models['mlp'+str(n1)+'-'+str(n2)+'-'+str(n3)] = MLPRegressor(solver='sgd', learning_rate_init=0.01, hidden_layer_sizes=(n1, n2, n3), verbose=False,  tol=0.00001, n_iter_no_change=1000, batch_size = 32, max_iter=100000)
	print('Defined %d models' % len(models))
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
#	steps.append(('standardize', StandardScaler()))
	# normalization
#	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# evaluate a single model
def evaluate_model(X, y, model, folds, metric):
	# create the pipeline
	pipeline = make_pipeline(model)
	# evaluate model
	scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)

	return scores

# evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, X_test, folds, metric):
	scores = None
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			scores = evaluate_model(X, y, model, folds, metric)
			fitModel = model.fit(X, y)
			predictions = model.predict(X_test)
	except:
		scores = None
		predictions = None
	return (scores, predictions)

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, X_test, folds=10, metric='accuracy'):
	results = dict()
	predicted = dict()
	for name, model in models.items():
		# evaluate the model
		scores, predictions = robust_evaluate_model(X, y, model, X_test, folds, metric)
		# show process
		if scores is not None:
			# store a result
			results[name] = scores
			predicted[name] = predictions
			mean_score, std_score, = mean(scores), std(scores)
			print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
		else:
			print('>%s: error' % name)
	return (results, predicted)

# print and plot the top n results
def summarize_results(results, predicted, y_test, thisCol, maximize=True, top_n=20):
	# check for no results
	if len(results) == 0:
		print('no results')
		return
	# determine how many results to summarize
	n = min(top_n, len(results))
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,mean(v)) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	# retrieve the top n for summarization
	names = [x[0] for x in mean_scores[:n]]
	scores = [results[x[0]] for x in mean_scores[:n]]
	predict = [predicted[x[0]] for x in mean_scores[:n]]
	# print the top n
	print()
	for i in range(n):
		name = names[i]
		mean_score, std_score = mean(results[name]), std(results[name])
		print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
	# boxplot for the top n
	plt.boxplot(scores, labels=names)
	_, labels = plt.xticks()
	plt.setp(labels, rotation=90)
	thisTitle = (thisCol+'_spotcheck.png')
	plt.savefig(thisTitle)
	f, axes = plt.subplots(4,5)
	for i in range(4):
		for j in range(5):
			axes[i,j].plot(y_test, predict[i*5+j], '.k')
			axes[i,j].set_title(names[i*5+j])
	f.suptitle(thisCol+'_Best_models_predictions')
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.savefig(thisCol+'_predictions.png')

	return scores

coltodrop = list(df_norm_scaled.iloc[:,0:214].columns)
dfdropped = df_norm_scaled.drop(columns = coltodrop)
colnames = list(dfdropped.columns)

for thisCol in colnames:
	#Withold a test set
	X_train, X_test, y_train, y_test = train_test_split(df_norm_scaled.iloc[:,0:213], df_norm_scaled[thisCol], test_size=0.2, random_state=42)
	index = y_train.index[y_train.apply(np.isnan)]
	todrop = index.values.tolist()
	X_train = X_train.drop(todrop)
	y_train = y_train.drop(todrop)
	index = y_test.index[y_test.apply(np.isnan)]
	todrop = index.values.tolist()
	X_test = X_test.drop(todrop)
	y_test = y_test.drop(todrop)
	# get model list
	models = get_models()
	# evaluate models
	results, predicted = evaluate_models(X_train, y_train, models, X_test, metric='neg_mean_squared_error')
	# summarize results
	summarize_results(results, predicted, y_test, thisCol)



###############################################################################








from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

X_train = df_norm_scaled.iloc[:,0:213]
rndperm = np.random.permutation(df.shape[0])

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_train.values)
X_train['pca-one'] = pca_result[:,0]
X_train['pca-two'] = pca_result[:,1]
X_train['pca-three'] = pca_result[:,2]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    xs=X_train.loc[rndperm,:]["pca-one"],
    ys=X_train.loc[rndperm,:]["pca-two"],
    zs=X_train.loc[rndperm,:]["pca-three"],

    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

umap = UMAP(n_components=3)


tsne = TSNE(n_components=3)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)







#######################


#########
# Multi-layer Perceptron Regressor (Neural Network)
#########
from sklearn.neural_network import MLPRegressor

for
lr = 0.01 #Learning rate
nn = [2, 3, 2, 2] #Neurons by layer

MLPr = MLPRegressor(solver='sgd', learning_rate_init=lr, hidden_layer_sizes=tuple(nn[1:]), verbose=True, n_iter_no_change=1000, batch_size = 64, max_iter=10000)
MLPr.fit(X_train, y_train)
y_pred = MLPr.predict(X_val)
plt.scatter(y_val, y_pred)

#MAE, MSE, RMSE
from sklearn import metrics
metrics.mean_absolute_error(y_val, y_pred)
metrics.mean_squared_error(y_val, y_pred)
np.sqrt(metrics.mean_squared_error(y_val, y_pred))


####################################################################

#Fine-tune the hyperparameters using cross-validation
#Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., should I replace missing values with zero or with the median value? Or just drop the rows?)
#Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimization approach
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(model, param_grid, verbose = 3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_

#Try Ensemble methods. Combining your best models will often perform better than running them individually
#Max Voting
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3 = LogisticRegression()

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)

pred1=model1.predict(X_test)
pred2=model2.predict(X_test)
pred3=model3.predict(X_test)

final_pred = np.array([])
for i in range(len(X_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))

#We can also use VotingClassifier from sklearn
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)

#Averaging
finalpred=(pred1+pred2+pred3)/3

#Weighted Average
finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)

#Stacking
from sklearn.model_selection import StratifiedKFold
def Stacking(model, train, y, test, n_fold):
	folds = StratifiedKFold(n_splits=n_fold, random_state=101)
	test_pred = np.empty((test.shape[0], 1), float)
	train_pred = np.empty((0, 1), float)
	for train_indices, val_indices in folds.split(train,y.values):
		X_train, X_val = train.iloc[train_indices], train.iloc[val_indices]
		y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

		model.fit(X_train, y_train)
		train_pred = np.append(train_pred, model.predict(X_val))
		test_pred = np.append(test_pred, model.predict(test))
	return test_pred.reshape(-1,1), train_pred

model1 = DecisionTreeClassifier(random_state=101)
test_pred1, train_pred1 = Stacking(model1, X_train, y_train, X_test, 10)
train_pred1 = pd.DataFrame(train_pred1)
test_pred1 = pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()
test_pred2, train_pred2 = Stacking(model2, X_train, y_train, X_test, 10)
train_pred2 = pd.DataFrame(train_pred2)
test_pred2 = pd.DataFrame(test_pred2)

df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

model = LogisticRegression(random_state=101)
model.fit(df,y_train)
model.score(df_test, y_test)

#Blending
model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
val_pred1 = pd.DataFrame(model1.predict(X_val))
test_pred1 = pd.DataFrame(model1.predict(X_test))

model2 = KNeighborsClassifier()
model2.fit(X_train,y_train)
val_pred2 = pd.DataFrame(model2.predict(X_val))
test_pred2 = pd.DataFrame(model2.predict(X_test))

df_val = pd.concat([X_val, val_pred1,val_pred2],axis=1)
df_test = pd.concat([X_test, test_pred1,test_pred2],axis=1)
model = LogisticRegression()
model.fit(df_val,y_val)
model.score(df_test,y_test)

#Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
ens = BaggingClassifier(DecisionTreeClassifier(random_state=101))
ens.fit(X_train, y_train)
ens.score(X_val,y_val)
#Regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
ens = BaggingRegressor(DecisionTreeRegressor(random_state=101))
ens.fit(X_train, y_train)
ens.score(X_val,y_val)

#Once you are confident about your final model, measure its performance on the test set to estimate the generalization error

#Model interpretability
#Feature importance
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=101).fit(X_val, y_val)
eli5.show_weights(perm, feature_names = X_val.columns.tolist())

#Partial dependence plot
#New integration in sklearn, might not work with older versions
from sklearn.inspection import partial_dependence, plot_partial_dependence
partial_dependence(model, X_train, features=['feature', ('feat1', 'feat2')])
plot_partial_dependence(model, X_train, features=['feature', ('feat1', 'feat2')])
#With external module for legacy editions
from pdpbox import pdp, get_dataset, info_plots

#Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=model, dataset=X_val, model_features=X_val.columns, feature='Goals Scored')

#plot it
pdp.pdp_plot(pdp_goals, 'Goals Scored')
plt.show()

#Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goals Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=model, dataset=X_val, model_features=X_val.columns, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

#ALE Plots: faster and unbiased alternative to partial dependence plots (PDPs). They have a serious problem when the features are correlated.
#The computation of a partial dependence plot for a feature that is strongly correlated with other features involves averaging predictions of artificial data instances that are unlikely in reality. This can greatly bias the estimated feature effect.
#https://github.com/blent-ai/ALEPython

#SHAP Values: Understand how each feature affects every individual prediciton
import shap
data_for_prediction = X_val.iloc[row_num]
explainer = shap.TreeExplainer(model)  #Use DeepExplainer for Deep Learning models, KernelExplainer for all other models
shap_vals = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_vals[1], data_for_prediction)

#We can also do a SHAP plot of the whole dataset
shap_vals = explainer.shap_values(X_val)
shap.summary_plot(shap_vals[1], X_val)
#SHAP Dependence plot
shap.dependence_plot('feature_for_x', shap_vals[1], X_val, interaction_index="feature_for_color")

#Local interpretable model-agnostic explanations (LIME)
#Surrogate models are trained to approximate the predictions of the underlying black box model. Instead of training a global surrogate model, LIME focuses on training local surrogate models to explain individual predictions.
#https://github.com/marcotcr/lime

#Dimensionality reduction
#SVD: Find the percentage of variance explained by each principal component
#First scale the data
U, S, V = np.linalg.svd(df, full_matrices=False)
importance = S/S.sum()
varinace_explained = importance.cumsum()*100
#PCA: Decompose the data in a defined number of variables keeping the most variance possible.
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(X_train_PCA)
X_train_PCA.index = X_train.index

X_test_PCA = pca.transform(X_test)
X_test_PCA = pd.DataFrame(X_test_PCA)
X_test_PCA.index = X_test.index

#ONLY FOR KAGGLE, NOT FOR REAL LIFE PROBLEMS
#If both train and test data come from the same distribution use this, we can use the target variable averaged over different categorical variables as a feature.
from sklearn import base
from sklearn.model_selection import KFold

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
	def __init__(self,colnames,targetName, n_fold=5, verbosity=True, discardOriginal_col=False):
		self.colnames = colnames
		self.targetName = targetName
		self.n_fold = n_fold
		self.verbosity = verbosity
		self.discardOriginal_col = discardOriginal_col

	def fit(self, X, y=None):
		return self

	def transform(self,X):
		assert(type(self.targetName) == str)
		assert(type(self.colnames) == str)
		assert(self.colnames in X.columns)
		assert(self.targetName in X.columns)

		mean_of_target = X[self.targetName].mean()
		kf = KFold(n_splits = self.n_fold, shuffle = True, random_state=2019)

		col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
		X[col_mean_name] = np.nan

		for tr_ind, val_ind in kf.split(X):
			X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
			X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
			X[col_mean_name].fillna(mean_of_target, inplace = True)

		if self.verbosity:
			encoded_feature = X[col_mean_name].values
			print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,
			np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))

		if self.discardOriginal_col:
			X = X.drop(self.targetName, axis=1)

		return X

targetc = KFoldTargetEncoderTrain('column','target',n_fold=5)
new_df = targetc.fit_transform(df)

new_df[['column_Kfold_Target_Enc','column']]