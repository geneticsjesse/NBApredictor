# ===================== #
# Classification models #
# ===================== #

# Source: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     April 4 2023

# How to run:   python3  stackingClassifier.py  -in merged_df_outliers_removed_CFS_Xscaled.csv
# ================= #
import argparse
import sys
import os
from sklearn import preprocessing
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
from xgboost import XGBClassifier
from numpy import arange
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# define command line arguments
parser = argparse.ArgumentParser(description='Classification Models')
parser.add_argument('--in_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file. The last colum of the file is the desired output.')
#parser.add_argument('--method', '-m', action="store", dest='method', default='linear', required=False, help='Method: adaboost, ridgec, dtc, gbc, knn, linear, linsvc, mlp, rf, sgd, svc')
#parser.add_argument('--kfold', '-k', action="store", dest='kfold', default=5, required=False, help='Number of folds for cross-validation')
#parser.add_argument('--num_splits', '-n', action="store", dest='num_splits', default=3, required=False, help='Number of folds for cross-validation')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename    = args.in_file

# Return: X and y vectors
def read_data():
    # load the dataset; header is first row
    df = pd.read_csv(filename, header=0)

    # separate input and output variables
    # remove non-integer columns to plot
    df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
# separate input and output variables
    varray  = df_slice.values
    nc      = len(varray[0,:])-1
    #X       = varray[:,9:]
    #Y       = varray [:,5]
    X       = varray[:,1:18]
    Y       = varray[:,nc]
    return X, Y

# define dataset
X, y = read_data()

# separate data into training/validation and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# get a stacking ensemble of models
def get_stacking():
 # define the base models
 level0 = list()
 level0.append(('lr', LogisticRegression()))#solver = 'lbfgs', max_iter=100000)))
 level0.append(('knn', KNeighborsClassifier()))#n_neighbors=5)))
 level0.append(('rf', RandomForestClassifier (random_state=0)))#n_estimators=100, random_state=0)))
 level0.append(('svm', SVC(gamma='auto')))#, C=1.0)))
 level0.append(('NB', GaussianNB()))
 level0.append(('mlp', MLPClassifier()))#(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))
 
 # define meta learner model
 level1 = LogisticRegression()
 # define the stacking ensemble
 model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
 return model

# initialize models
models = []
models.append(('lr', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append (('knn', KNeighborsClassifier()))#n_neighbors=5)))
models.append(('rf', RandomForestClassifier (random_state=0)))#n_estimators=100, random_state=0)))
models.append (('svm', SVC(gamma='auto')))#, C=1.0)))
models.append(('nb', GaussianNB()))
models.append(('mlp', MLPClassifier()))#solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))
models.append(('stacking', get_stacking()))

# models['lr'] = LogisticRegression(solver = 'lbfgs', max_iter=100000)
# models['knn'] = KNeighborsClassifier(n_neighbors=5)
# models['rf'] = RandomForestClassifier (n_estimators=100, random_state=0)
# models['svm'] = SVC(C=1.0, gamma='auto')
# models['NB'] = GaussianNB()
# models['mlp'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#models['stacking'] = get_stacking()
# return models

# evaluate a given model using cross-validation
print('\nModel evalution - training')
print('--------------------------')
results = []
names = []
for name, model in models:
        # FIX N_REPEATS TO 3 NOT 1
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='matthews_corrcoef')
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
# Increase C here? Tuning HP seems to select optimal close to 2
model_params['lr']['C'] = list(arange(0.1,2,0.1))
model_params['knn'] = dict()
# Increase n_neighbours? Optimal =19 
model_params['knn']['n_neighbors'] = list(range(1,21,2))
model_params['rf'] = dict()
# Increase range here?
model_params['rf']['n_estimators']=list(range(0,100,10))
model_params['nb'] = dict()
model_params['nb']['var_smoothing'] = list(arange(1e-10,1e-08,1e-9))
model_params['svm'] = dict()
model_params['svm']['C'] = list(arange(0.01,1.6,0.2))
model_params['mlp'] = dict()
model_params['mlp']['activation'] = ['tanh', 'relu']
model_params['mlp']['hidden_layer_sizes'] = [(50,50,50), (50,100,50), (100,)]
model_params['mlp']['solver'] = ['sgd', 'adam']
model_params['mlp']['alpha'] = list(arange(0.0001, 0.001, 0.0001))
model_params['mlp']['learning_rate'] = ['constant','adaptive']
model_params['stacking'] = dict()
model_params['stacking']['cv'] = list(arange(0,10,2))
model_params['stacking']['final_estimator'] = [LogisticRegression(max_iter=100000), RandomForestClassifier()]#, KNeighborsClassifier()], XGBClassifier()]

best_params = dict()
for name, model in models:
    # FIX N_REPEATS =3 NOT 1
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
    # FIX N_ITER TO 5 NOT 1
    rand_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=1, n_jobs=-1, cv=cv, scoring='matthews_corrcoef')
    rand_result = rand_search.fit(X_train, Y_train)
    print("Model %s -- Best: %f using %s" % (name, rand_result.best_score_, rand_result.best_params_))
    best_params[name] = rand_result.best_params_

# re-initialize models using best parameter settings
optimized_models = []
optimized_models.append(('lr', LogisticRegression(solver='liblinear', multi_class='ovr', C=best_params['lr']['C'])))
optimized_models.append(('rf', RandomForestClassifier(n_estimators=best_params['rf']['n_estimators'])))
optimized_models.append(('knn', KNeighborsClassifier(n_neighbors=best_params['knn']['n_neighbors'])))
optimized_models.append(('nb', GaussianNB(var_smoothing=best_params['nb']['var_smoothing'])))
optimized_models.append(('svm', SVC(gamma='auto',C=best_params['svm']['C'])))
optimized_models.append(('mlp', MLPClassifier (activation=best_params['mlp']['activation'], hidden_layer_sizes=best_params['mlp']['hidden_layer_sizes'],
                                                solver = best_params['mlp']['solver'], alpha = best_params['mlp']['alpha'], learning_rate=best_params['mlp']['learning_rate'])))
optimized_models.append(('stacking', StackingClassifier(estimators= [('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('rf', RandomForestClassifier(random_state=0)), ('svm', SVC(gamma='auto')), ('NB', GaussianNB()), ('mlp', MLPClassifier())], cv = best_params['stacking']['cv'], final_estimator=best_params['stacking']['final_estimator'])))

print('\nModel evalution - optimized')
print('--------------------------')
results = []
names = []
for name, model in optimized_models:
        # FIX N_REPEATS =3 NOT 1
	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='matthews_corrcoef')
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
	model.fit(X_train, Y_train)
	predicted_results = model.predict(X_test)
	acc_result = matthews_corrcoef(predicted_results,Y_test)
	print('%s Matthews Correlation Coefficient: %f' % (name, acc_result))
	cm = confusion_matrix(predicted_results,Y_test)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot()
	plt.title('Confusion matrix corresp. to test results for ' + name)
	plt.xlabel('Ground truth')
	plt.ylabel('Predicted results')

	plt.show()