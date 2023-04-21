# ===================== #
# Running classification models and hyperparameter optimization #
# ===================== #

# Source: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     April 4 2023

# How to run:   python3  stackingClassifier_2.py  --in_trainfile train2015_2021_RFEcommon --in_testfile test_RFEcommon.csv
# This script runs our 7 models on training data, optimizes hyperparameters, and ultimately runs the models on testing data to evaluate model performance.
# ================= #

# Import relevant libraries
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.ensemble import StackingClassifier
from numpy import arange
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import researchpy as rp

print ("\nBeginning stackingClassifier.py. This one might take a while (5-10 minutes)\n")

##################
# set font sizes #
##################
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Make directory if does not exist
path = "hyperparameterOptimization"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

# Make directory if does not exist
path = "confusionMatrices"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

# Make directory if does not exist
path = "optimizedModels"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)   

# Return: X and y vectors
def read_train():
    # load the dataset; header is first row
    df = pd.read_csv('./RFE_splits/train2015_2021_RFEcommon.csv', header=0)

    # separate input and output variables
    # remove non-integer columns to plot
    df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
# separate input and output variables
    varray  = df_slice.values
    nc      = len(varray[0,:])-1
    X       = varray[:,0:nc] 
    Y       = varray[:,nc]
    return X, Y

# define dataset
X_train, y_train = read_train()

# Return: X and y vectors
def read_test():
    # load the dataset; header is first row
    df = pd.read_csv('./RFE_splits/test_RFEcommon.csv', header=0)

    # separate input and output variables
    # remove non-integer columns to plot
    df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
# separate input and output variables
    varray  = df_slice.values
    nc      = len(varray[0,:])-1
    X       = varray[:,0:nc]
    Y       = varray[:,nc]
    return X, Y

# define dataset
X_test, y_test = read_test()

# get a stacking ensemble of models
def get_stacking():
 level0 = list()
 level0.append(('LR', LogisticRegression(max_iter=10000000, random_state=2)))
 level0.append(('KNN', KNeighborsClassifier()))
 level0.append(('RF', RandomForestClassifier (random_state=2)))
 level0.append(('SVC', SVC(gamma='auto', random_state=2)))
 level0.append(('NB', GaussianNB()))
 level0.append(('MLP', MLPClassifier(random_state=2)))
 
 # define meta learner model
 level1 = LogisticRegression()
 # define the stacking ensemble
 model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
 return model

# initialize models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)))
models.append (('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier (random_state=0)))
models.append (('SVC', SVC(gamma='auto', random_state=0)))
models.append(('NB', GaussianNB()))
models.append(('MLP', MLPClassifier(random_state=0)))
models.append(('Stacking', get_stacking()))

# evaluate a given model using cross-validation
print('\nModel evalution - training')
print('--------------------------')
results = []
names = []
for name, model in models:
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='matthews_corrcoef')
	results.append(cv_results)
	names.append(name)
	print('>%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

fig1,ax1 = plt.subplots(figsize = (10, 7))
plot1=ax1.boxplot(results, labels=names, showmeans=True)
ax1.set_title('Algorithm Comparison - before optimization')
ax1.set_ylabel('Matthews Correlation Coefficient')
fig1.savefig(f"./hyperparameterOptimization/modelComparison_beforeOptimization.png")
plt.close(fig1)
#plt.show()

#improve accuracy with hyper-parameter tuning
print('\nModel evaluation - hyper-parameter tuning')
print('-----------------------------------------')

model_params = dict()
model_params['LR'] = dict()
model_params['LR']['C'] = list(arange(0.1,2,0.1))
model_params['LR']['class_weight']=['balanced']
model_params['LR']['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
model_params['LR']['max_iter']=list(arange(100,10000000, 100))
model_params['KNN'] = dict()
model_params['KNN']['n_neighbors'] = list(arange(1,30,2))
model_params['KNN']['leaf_size'] = list(arange(10,50,10))
model_params['KNN']['metric'] = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'braycurtis']
model_params['RF'] = dict()
model_params['RF']['n_estimators']=list(arange(1,1000,100))
model_params['RF']['max_samples']=list(arange(0.1,0.9,0.1))
model_params['RF']['max_depth']=list(arange(1,26,1))
model_params['RF']['criterion']=['gini', 'entropy']
model_params['RF']['max_features']=list(arange(1,17,1))
model_params['NB'] = dict()
model_params['NB']['var_smoothing'] = list(arange(1e-10,1e-08,1e-9))
model_params['SVC'] = dict()
model_params['SVC']['C'] = list(arange(0.01,2.0,0.2))
model_params['SVC']['kernel'] = ['poly', 'rbf', 'sigmoid']
model_params['SVC']['class_weight']=['balanced']
model_params['SVC']['degree'] = list(arange(1,3,1))
model_params['MLP'] = dict()
model_params['MLP']['activation'] = ['tanh', 'relu']
model_params['MLP']['hidden_layer_sizes'] = [(50,50,50), (50,100,50), (100,)]
model_params['MLP']['solver'] = ['sgd', 'adam']
model_params['MLP']['alpha'] = list(arange(0.0001, 0.001, 0.0001))
model_params['MLP']['learning_rate'] = ['constant','adaptive']
model_params['Stacking'] = dict()
model_params['Stacking']['cv'] = list(arange(0,10,2))
model_params['Stacking']['final_estimator'] = [LogisticRegression(max_iter=10000000), SVC()]

best_params = dict()
for name, model in models:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    rand_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=5, n_jobs=-1, cv=cv, scoring='matthews_corrcoef')
    rand_result = rand_search.fit(X_train, y_train)
    print("Model %s -- Best: %f using %s" % (name, rand_result.best_score_, rand_result.best_params_))
    best_params[name] = rand_result.best_params_

# re-initialize models using best parameter settings
optimized_models = []
optimized_models.append(('LR', LogisticRegression(multi_class='ovr', C=best_params['LR']['C'], class_weight=best_params['LR']['class_weight'], solver=best_params['LR']['solver'], max_iter=best_params['LR']['max_iter'], random_state=2)))
optimized_models.append(('KNN', KNeighborsClassifier(n_neighbors=best_params['KNN']['n_neighbors'])))
optimized_models.append(('RF', RandomForestClassifier(n_estimators=best_params['RF']['n_estimators'], max_samples=best_params['RF']['max_samples'], max_depth=best_params['RF']['max_depth'], criterion=best_params['RF']['criterion'], random_state=2)))
optimized_models.append(('SVC', SVC(gamma='auto',C=best_params['SVC']['C'], kernel=best_params['SVC']['kernel'], class_weight=best_params['SVC']['class_weight'], degree=best_params['SVC']['degree'], random_state=2)))
optimized_models.append(('NB', GaussianNB(var_smoothing=best_params['NB']['var_smoothing'])))
optimized_models.append(('MLP', MLPClassifier (activation=best_params['MLP']['activation'], hidden_layer_sizes=best_params['MLP']['hidden_layer_sizes'], solver = best_params['MLP']['solver'], alpha = best_params['MLP']['alpha'], learning_rate=best_params['MLP']['learning_rate'], learning_rate_init = 1e-05, max_iter=100000, random_state=2)))

# create a list of tuples containing the optimized models and their names to use for appending our stacking classifier to the optimized models list
optimized_models_forStacking = [('LR', LogisticRegression(C=best_params['LR']['C'], class_weight=best_params['LR']['class_weight'], max_iter=best_params['LR']['max_iter'], multi_class='ovr', random_state=2, solver=best_params['LR']['solver'])),
    ('RF', RandomForestClassifier(criterion=best_params['RF']['criterion'], max_depth=best_params['RF']['max_depth'], max_samples=best_params['RF']['max_samples'], n_estimators=best_params['RF']['n_estimators'], random_state=2)),
    ('KNN', KNeighborsClassifier(n_neighbors=best_params['KNN']['n_neighbors'])),
    ('SVC', SVC(C=best_params['SVC']['C'], class_weight=best_params['SVC']['class_weight'], degree=best_params['SVC']['degree'], gamma='auto', kernel=best_params['SVC']['kernel'], random_state=2)),
    ('NB', GaussianNB(var_smoothing=best_params['NB']['var_smoothing'])),
    ('MLP', MLPClassifier(activation=best_params['MLP']['activation'], alpha=best_params['MLP']['alpha'], hidden_layer_sizes=best_params['MLP']['hidden_layer_sizes'], learning_rate=best_params['MLP']['learning_rate'], max_iter=10000, learning_rate_init = 1e-05, random_state=2, solver=best_params['MLP']['solver']))
]
# Need to append stacking to optimized models
optimized_models.append(('Stacking', StackingClassifier(estimators= optimized_models_forStacking, cv = best_params['Stacking']['cv'], final_estimator=best_params['Stacking']['final_estimator'])))

print('\nModel evalution - optimized')
print('--------------------------')
results = []
names = []
for name, model in optimized_models:
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='matthews_corrcoef')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare optimized models based on training results
fig2,ax2 = plt.subplots(figsize = (10, 7))
plot2=ax2.boxplot(results, labels=names, showmeans=True)
ax2.set_title('Algorithm Comparison - after optimization')
ax2.set_ylabel('Matthews Correlation Coefficient')
fig2.savefig(f"./hyperparameterOptimization/modelComparison_afterOptimization.png")
plt.close(fig2)
#plt.show()

# Compute a paired t-test to determine if the model hyperparameters are significantly different before and after optimization.
print ('\n Computing a paired t-test to determine if the model hyperparameters are significantly different before and after optimization.')
print('--------------------------')
before_opt = pd.Series([0.267966,0.229173,0.234368,0.262041,0.258832,0.203068 ,0.255868])
after_opt = pd.Series([0.267966,0.227546,0.235536,0.258832,0.262041,0.245072,0.251862])

summary, results = rp.ttest(before_opt, after_opt, paired=True)
print("The results of the t-test are below\n\n", results)

# fit and save optimized models
for name, model in optimized_models:
	model.fit(X_train, y_train)
	filename = (f'./optimizedModels/{name}_optimized_model.sav')
	joblib.dump(model, filename)

# testing
print('\nModel testing')
print('-------------')
for name, model in optimized_models:
    model.fit(X_train, y_train)
    predicted_results = model.predict(X_test)
    acc_result = matthews_corrcoef(predicted_results,y_test)
    print('%s Matthews Correlation Coefficient: %f' % (name, acc_result))
    cm = confusion_matrix(predicted_results,y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix corresp. to test results for ' + name)
    plt.xlabel('Ground truth')
    plt.ylabel('Predicted results')
    plt.savefig(f'confusionMatrices/{name}_ConfusionMatrix.png')

print ("stackingClassifier.py has finished running. The program is complete. Best of luck with your sports betting! ;-)\n")