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
from sklearn.metrics import matthews_corrcoef, accuracy_score
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
    X       = varray[:,0:nc] #0-18 typically
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
 level0.append(('lr', LogisticRegression(max_iter=10000000, random_state=2)))
 level0.append(('knn', KNeighborsClassifier()))
 level0.append(('rf', RandomForestClassifier (random_state=2)))
 level0.append(('svm', SVC(gamma='auto', random_state=2)))
 level0.append(('NB', GaussianNB()))
 level0.append(('mlp', MLPClassifier(random_state=2)))
 
 # define meta learner model
 level1 = LogisticRegression()
 # define the stacking ensemble
 model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
 return model

# initialize models
models = []
models.append(('lr', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)))
models.append (('knn', KNeighborsClassifier()))
models.append(('rf', RandomForestClassifier (random_state=0)))
models.append (('svm', SVC(gamma='auto', random_state=0)))
models.append(('nb', GaussianNB()))
models.append(('mlp', MLPClassifier(random_state=0)))
models.append(('stacking', get_stacking()))

# evaluate a given model using cross-validation
print('\nModel evalution - training')
print('--------------------------')
results = []
names = []
for name, model in models:
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('>%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# fig1,ax1 = plt.subplots(figsize = (10, 7))
# plot1=ax1.boxplot(results, labels=names, showmeans=True)
# ax1.set_title('Algorithm Comparison - before optimization')
# ax1.set_ylabel('Matthews Correlation Coefficient')
# fig1.savefig(f"./hyperparameterOptimization/modelComparison_beforeOptimization_mlptest.png")
# plt.close(fig1)
#plt.show()

#improve accuracy with hyper-parameter tuning
print('\nModel evaluation - hyper-parameter tuning')
print('-----------------------------------------')

model_params = dict()
model_params['lr'] = dict()
model_params['lr']['C'] = list(arange(0.1,2,0.1))
model_params['lr']['class_weight']=['balanced']
model_params['lr']['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
model_params['lr']['max_iter']=list(arange(100,10000000, 100))
model_params['knn'] = dict()
model_params['knn']['n_neighbors'] = list(arange(1,30,2))
model_params['knn']['leaf_size'] = list(arange(10,50,10))
model_params['knn']['metric'] = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'braycurtis']
model_params['rf'] = dict()
model_params['rf']['n_estimators']=list(arange(1,1000,100))
model_params['rf']['max_samples']=list(arange(0.1,0.9,0.1))
model_params['rf']['max_depth']=list(arange(1,26,1))
model_params['rf']['criterion']=['gini', 'entropy']
model_params['rf']['max_features']=list(arange(1,17,1))
model_params['nb'] = dict()
model_params['nb']['var_smoothing'] = list(arange(1e-10,1e-08,1e-9))
model_params['svm'] = dict()
model_params['svm']['C'] = list(arange(0.01,2.0,0.2))
model_params['svm']['kernel'] = ['poly', 'rbf', 'sigmoid']
model_params['svm']['class_weight']=['balanced']
model_params['svm']['degree'] = list(arange(1,3,1))
model_params['mlp'] = dict()
model_params['mlp']['activation'] = ['tanh', 'relu']
model_params['mlp']['hidden_layer_sizes'] = [(50,50,50), (50,100,50), (100,)]
model_params['mlp']['solver'] = ['sgd', 'adam']
model_params['mlp']['alpha'] = list(arange(0.0001, 0.001, 0.0001))
model_params['mlp']['learning_rate'] = ['constant','adaptive']
model_params['stacking'] = dict()
model_params['stacking']['cv'] = list(arange(0,10,2))
model_params['stacking']['final_estimator'] = [LogisticRegression(max_iter=10000000), SVC()]

best_params = dict()
for name, model in models:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    rand_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=5, n_jobs=-1, cv=cv, scoring='accuracy')
    rand_result = rand_search.fit(X_train, y_train)
    print("Model %s -- Best: %f using %s" % (name, rand_result.best_score_, rand_result.best_params_))
    best_params[name] = rand_result.best_params_

# re-initialize models using best parameter settings
optimized_models = []
optimized_models.append(('lr', LogisticRegression(multi_class='ovr', C=best_params['lr']['C'], class_weight=best_params['lr']['class_weight'], solver=best_params['lr']['solver'], max_iter=best_params['lr']['max_iter'], random_state=2)))
optimized_models.append(('rf', RandomForestClassifier(n_estimators=best_params['rf']['n_estimators'], max_samples=best_params['rf']['max_samples'], max_depth=best_params['rf']['max_depth'], criterion=best_params['rf']['criterion'], random_state=2)))
optimized_models.append(('knn', KNeighborsClassifier(n_neighbors=best_params['knn']['n_neighbors'])))
optimized_models.append(('nb', GaussianNB(var_smoothing=best_params['nb']['var_smoothing'])))
optimized_models.append(('svm', SVC(gamma='auto',C=best_params['svm']['C'], kernel=best_params['svm']['kernel'], class_weight=best_params['svm']['class_weight'], degree=best_params['svm']['degree'], random_state=2)))
optimized_models.append(('mlp', MLPClassifier (activation=best_params['mlp']['activation'], hidden_layer_sizes=best_params['mlp']['hidden_layer_sizes'], solver = best_params['mlp']['solver'], alpha = best_params['mlp']['alpha'], learning_rate=best_params['mlp']['learning_rate'], learning_rate_init = 1e-05, max_iter=100000, random_state=2)))

# create a list of tuples containing the optimized models and their names to use for appending our stacking classifier to the optimized models list
optimized_models_forStacking = [('lr', LogisticRegression(C=best_params['lr']['C'], class_weight=best_params['lr']['class_weight'], max_iter=best_params['lr']['max_iter'], multi_class='ovr', random_state=2, solver=best_params['lr']['solver'])),
    ('rf', RandomForestClassifier(criterion=best_params['rf']['criterion'], max_depth=best_params['rf']['max_depth'], max_samples=best_params['rf']['max_samples'], n_estimators=best_params['rf']['n_estimators'], random_state=2)),
    ('knn', KNeighborsClassifier(n_neighbors=best_params['knn']['n_neighbors'])),
    ('nb', GaussianNB(var_smoothing=best_params['nb']['var_smoothing'])),
    ('svm', SVC(C=best_params['svm']['C'], class_weight=best_params['svm']['class_weight'], degree=best_params['svm']['degree'], gamma='auto', kernel=best_params['svm']['kernel'], random_state=2)),
    ('mlp', MLPClassifier(activation=best_params['mlp']['activation'], alpha=best_params['mlp']['alpha'], hidden_layer_sizes=best_params['mlp']['hidden_layer_sizes'], learning_rate=best_params['mlp']['learning_rate'], max_iter=10000, learning_rate_init = 1e-05, random_state=2, solver=best_params['mlp']['solver']))
]
# Need to append stacking to optimized models
optimized_models.append(('stacking', StackingClassifier(estimators= optimized_models_forStacking, cv = best_params['stacking']['cv'], final_estimator=best_params['stacking']['final_estimator'])))

print('\nModel evalution - optimized')
print('--------------------------')
results = []
names = []
for name, model in optimized_models:
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare optimized models based on training results
# fig2,ax2 = plt.subplots(figsize = (10, 7))
# plot2=ax2.boxplot(results, labels=names, showmeans=True)
# ax2.set_title('Algorithm Comparison - after optimization')
# ax2.set_ylabel('Matthews Correlation Coefficient')
# fig2.savefig(f"./hyperparameterOptimization/modelComparison_afterOptimization_mlptest.png")
# plt.close(fig2)
# #plt.show()

# fit and save optimized models
# for name, model in optimized_models:
# 	model.fit(X_train, y_train)
# 	filename = (f'./optimizedModels/{name}_optimized_model.sav')
# 	joblib.dump(model, filename)

# testing
print('\nModel testing')
print('-------------')
for name, model in optimized_models:
    model.fit(X_train, y_train)
    predicted_results = model.predict(X_test)
    acc_result = accuracy_score(predicted_results,y_test)
    print('%s Accuracy: %f' % (name, acc_result))
    # cm = confusion_matrix(predicted_results,y_test)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.title('Confusion matrix corresp. to test results for ' + name)
    # plt.xlabel('Ground truth')
    # plt.ylabel('Predicted results')
    # plt.savefig(f'confusionMatrices/{name}_ConfusionMatrix_mlptest.png')