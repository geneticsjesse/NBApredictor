# ===================== #
# Classification models #
# ===================== #
# Author:   Dan Tulpan, dtulpan@uoguelph.ca
# Date:     March 16, 2021

# How to run:   python3  stackingClassifier.py  -in merged_df_outliers_removed_CFS.csv -m linear  -k 10  -n 3
# ================= #
import argparse
import sys
import os
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
import pandas as pd
from sklearn.ensemble import StackingClassifier
from numpy import mean, std

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
    X       = varray[:,1:18]
    Y       = varray[:,nc]
    return X, Y

# get a stacking ensemble of models
def get_stacking():
 # define the base models
 level0 = list()
 level0.append(('lr', LogisticRegression(solver = 'lbfgs', max_iter=100000)))
 level0.append(('knn', KNeighborsClassifier(n_neighbors=5)))
 level0.append(('rf', RandomForestClassifier (n_estimators=100, random_state=0)))
 level0.append(('svm', SVC(C=1.0, gamma='auto')))
 level0.append(('bayes', GaussianNB()))
 level0.append(('mlp', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))

 # define meta learner model
 level1 = LogisticRegression()
 # define the stacking ensemble
 model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
 return model
# get a list of models to evaluate
def get_models():
 models = dict()
 models['lr'] = LogisticRegression(solver = 'lbfgs', max_iter=100000)
 models['knn'] = KNeighborsClassifier(n_neighbors=5)
 models['rf'] = RandomForestClassifier (n_estimators=100, random_state=0)
 models['svm'] = SVC(C=1.0, gamma='auto')
 models['bayes'] = GaussianNB()
 models['mlp'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
 models['stacking'] = get_stacking()
 return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 #Y_pred = cross_val_predict (model, X, y, cv = cv)
 #scores = matthews_corrcoef (y, Y_pred)
 scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
 return scores

# define dataset
X, y = read_data()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 scores = evaluate_model(model, X, y)
 results.append(scores)
 names.append(name)
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()
