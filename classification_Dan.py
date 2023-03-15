# ===================== #
# Classification models #
# ===================== #
# Author:   Dan Tulpan, dtulpan@uoguelph.ca
# Date:     March 16, 2021

# How to run:   python3  classification.py  -in iris_num.csv  -m linear  -k 10  -n 3
# ================= #

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, RepeatedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import model_selection
import argparse
import sys
import os
import pandas as pd


method_name = {
    "adaboost": "AdaBoost Classification",
    "ridgec": "Ridge Classification",
    "dtc": "Decision Tree Classification",
    "gbc": "Gradient Boosting Classification",
    "knn": "K-Nearest Neighbour Classification",
    "linear": "Linear Classification",
    "linsvc": "Linear Support Vector Classification",
    "mlp": "Multi-Layer Perceptron Classification",
    "rf": "Random Forest Classification",
    "sgd": "Stochastic Gradient Descent Classification",
    "svc": "Support Vector Classification"
}

scoring = { 'Accuracy':'accuracy',
            'Balanced accuracy':'balanced_accuracy',
            'Precision':'precision_macro',
            'Recall':'recall_macro',
            'F1-score':'f1_macro'
        }

# Set classifier model
# Return: regressor and method name (for printing purposes)
def set_classifier(method):
    if (method == "adaboost"):
        classifier = AdaBoostClassifier(random_state=0, n_estimators=100)

    elif (method == "ridgec"):
        classifier = RidgeClassifier()

    elif (method == "dtc"):
        classifier = tree.DecisionTreeClassifier()

    elif (method == "gbc"):
        classifier = GradientBoostingClassifier(n_estimators=100)

    elif (method == "knn"):
        classifier = KNeighborsClassifier(n_neighbors=5)

    elif (method == "linear"):
        classifier = LogisticRegression(max_iter=100000)

    elif (method == "linsvc"):
        classifier = LinearSVC(random_state=0, tol=1e-05, max_iter=100000)

    # sometimes throws warnings for labels/classes with no predicted samples
    elif (method == "mlp"):
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    elif (method == "rf"):
        classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    elif (method == "sgd"):
        classifier = SGDClassifier(max_iter=100000, tol=1e-3)

    elif (method == "svc"):
        classifier = SVC(C=1.0, gamma='auto')

    else:
        print("\nError: Invalid method name:" + method + "\n")
        parser.print_help()
        sys.exit(0)
    return classifier, method_name[method]

# Evaluate classifier using K-fold cross validation
def eval_model(classifier, num_sp, num_rep):
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    #cv = ShuffleSplit(n_splits=num_sp, test_size=0.2, random_state=1)
    kfold = RepeatedKFold(n_splits=num_sp, n_repeats=num_rep, random_state=1)

    num_characters = 20
    print("Model".ljust(num_characters),":", method_name)
    print("K-folds".ljust(num_characters),":", kf)
    print("Num splits".ljust(num_characters),":", num_splits)

    for name,score in scoring.items():
        results = model_selection.cross_val_score(classifier, X, y, cv=kfold, scoring=score, n_jobs=-1)
        print(name.ljust(num_characters), ": %.3f (%.3f)" % (np.absolute(results.mean()), np.absolute(results.std())))

# Plot predicted values against true values
def plot_predictions(classifier, num_sp):
    predicted = cross_val_predict(classifier, X, y, cv=num_sp, n_jobs=-1)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

# Read data from csv file
# Assumptions:  last column in the file represents the predictor/dependent variable
#               all data is numeric and has been properly pre-processed
# Return: X and y vectors
def read_data():
    # load the dataset; header is first row
    df = pd.read_csv(filename, header=0)

    # separate input and output variables
    varray  = df.values
    nc      = len(varray[0,:])-1
    X       = varray[:,0:nc]
    y       = varray[:,nc]
    return X, y

# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------
# define command line arguments
parser = argparse.ArgumentParser(description='Classification Models')
parser.add_argument('--in_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file. The last colum of the file is the desired output.')
parser.add_argument('--method', '-m', action="store", dest='method', default='linear', required=False, help='Method: adaboost, ridgec, dtc, gbc, knn, linear, linsvc, mlp, rf, sgd, svc')
parser.add_argument('--kfold', '-k', action="store", dest='kfold', default=5, required=False, help='Number of folds for cross-validation')
parser.add_argument('--num_splits', '-n', action="store", dest='num_splits', default=3, required=False, help='Number of folds for cross-validation')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename    = args.in_file
method      = args.method
kf          = int(args.kfold)
num_splits  = int(args.num_splits)

# set regressor based on user choice
classifier, method_name = set_classifier(method)

# load data from file
X, y = read_data()

# evaluate model
eval_model(classifier, kf, num_splits)

# plot predicted values
plot_predictions(classifier, kf)
