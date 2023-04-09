# ===================== #
# Classification models #
# ===================== #

# Source: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     April 4 2023

# How to run:   python3  stackingClassifier_2.py  --in_trainfile training2015-2021.csv_outliers_removed_scaled_RFECOPY_JW.csv --in_testfile test_set_outliers_removed_scaled.csv
# ================= #
import argparse
import sys
import os
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
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

# define command line arguments
parser = argparse.ArgumentParser(description='Classification Models')
parser.add_argument('--in_trainfile', '-train', action="store", dest='in_trainfile', required=True, help='Name of training csv input file.')
parser.add_argument('--in_testfile', '-test', action="store", dest='in_testfile', required=True, help='Name of testing csv input file. The last colum of the file is the desired output.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename_train    = args.in_trainfile
filename_test = args.in_testfile

# Return: X and y vectors
def read_train():
    # load the dataset; header is first row
    df = pd.read_csv(filename_train, header=0)

    # separate input and output variables
    # remove non-integer columns to plot
    df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
# separate input and output variables
    varray  = df_slice.values
    nc      = len(varray[0,:])-1
    #X       = varray[:,9:]
    #Y       = varray [:,5]
    X       = varray[:,0:9] #0-18 typically
    Y       = varray[:,nc]
    return X, Y

# define dataset
X_train, y_train = read_train()

# Return: X and y vectors
def read_test():
    # load the dataset; header is first row
    df = pd.read_csv(filename_test, header=0)

    # separate input and output variables
    # remove non-integer columns to plot
    df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
# separate input and output variables
    varray  = df_slice.values
    nc      = len(varray[0,:])-1
    #X       = varray[:,9:]
    #Y       = varray [:,5]
    X       = varray[:,0:9] #0-18 typically
    Y       = varray[:,nc]
    return X, Y

# define dataset
X_test, y_test = read_test()

# separate data into training/validation and testing datasets
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# get a stacking ensemble of models
def get_stacking():
 # define the base models - UPDATE THESE WITH TUNED HP 
 level0 = list()
 level0.append(('lr', LogisticRegression(max_iter=1000000, random_state=2)))
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
models.append(('mlp', MLPClassifier(random_state=0)))#solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))
models.append(('stacking', get_stacking()))

# evaluate a given model using cross-validation
print('\nModel evalution - training')
print('--------------------------')
results = []
names = []
for name, model in models:
        # FIX N_REPEATS TO 3 NOT 1
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='matthews_corrcoef')
	results.append(cv_results)
	names.append(name)
	print('>%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names, showmeans=True)
plt.title('Algorithm Comparison - before optimization')
plt.show()

#improve accuracy with hyper-parameter tuning
print('\nModel evaluation - hyper-parameter tuning')
print('-----------------------------------------')

model_params = dict()
model_params['lr'] = dict()
model_params['lr']['C'] = list(arange(0.1,2,0.1))
model_params['lr']['class_weight']=['balanced']
model_params['lr']['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
model_params['lr']['max_iter']=list(arange(100,1000000, 100))
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
    # FIX N_REPEATS =3 NOT 1
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
    # FIX N_ITER TO 5 NOT 1
    #grid_search = GridSearchCV(estimator=model, param_grid=model_params[model], cv =cv, n_jobs=4, scoring= "matthews_corrcoef", refit=True, return_train_score=False)
    #grid_result = grid_search.fit(X_train, Y_train)
    rand_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=1, n_jobs=-1, cv=cv, scoring='matthews_corrcoef')
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
optimized_models.append(('mlp', MLPClassifier (activation=best_params['mlp']['activation'], hidden_layer_sizes=best_params['mlp']['hidden_layer_sizes'], solver = best_params['mlp']['solver'], alpha = best_params['mlp']['alpha'], learning_rate=best_params['mlp']['learning_rate'], max_iter=10000, random_state=2)))

# create a list of tuples containing the optimized models and their names
optimized_models_forStacking = [('lr', LogisticRegression(C=best_params['lr']['C'], class_weight=best_params['lr']['class_weight'], max_iter=best_params['lr']['max_iter'], multi_class='ovr', random_state=2, solver=best_params['lr']['solver'])),
    ('rf', RandomForestClassifier(criterion=best_params['rf']['criterion'], max_depth=best_params['rf']['max_depth'], max_samples=best_params['rf']['max_samples'], n_estimators=best_params['rf']['n_estimators'], random_state=2)),
    ('knn', KNeighborsClassifier(n_neighbors=best_params['knn']['n_neighbors'])),
    ('nb', GaussianNB(var_smoothing=best_params['nb']['var_smoothing'])),
    ('svm', SVC(C=best_params['svm']['C'], class_weight=best_params['svm']['class_weight'], degree=best_params['svm']['degree'], gamma='auto', kernel=best_params['svm']['kernel'], random_state=2)),
    ('mlp', MLPClassifier(activation=best_params['mlp']['activation'], alpha=best_params['mlp']['alpha'], hidden_layer_sizes=best_params['mlp']['hidden_layer_sizes'], learning_rate=best_params['mlp']['learning_rate'], max_iter=10000, random_state=2, solver=best_params['mlp']['solver']))
]

# Need to append stacking to optimized models
optimized_models.append(('stacking', StackingClassifier(estimators= optimized_models_forStacking, cv = best_params['stacking']['cv'], final_estimator=best_params['stacking']['final_estimator'])))

print('\nModel evalution - optimized')
print('--------------------------')
results = []
names = []
for name, model in optimized_models:
        # FIX N_REPEATS =3 NOT 1
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='matthews_corrcoef')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare optimized models based on training results
plt.boxplot(results, labels=names, showmeans=True)
plt.title('Algorithm Comparison - after optimization')
plt.show()

# # fit and save optimized models
# for name, model in optimized_models:
# 	model.fit(X_train, Y_train)
# 	filename = name + '_optimized_model.sav'
# 	joblib.dump(model, filename)

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

	plt.show()
