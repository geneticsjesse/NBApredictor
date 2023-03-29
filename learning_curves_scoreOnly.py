# ========================================================================= #
# Learning curves for classification (comparison between two methods)
# reference and inspiration
# source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 26, 2023
#
# How to run:   python3 learning_curves_scoreOnly.py -in merged_df_outliers_removed_CFS.csv -m1 ann -m2 gnb -m3 rf -m4 knn -m5 linear -m6 svc -n 10  
# ========================================================================= #
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import argparse
import sys
import os
import pandas as pd

# Set classifier model
# Return classifier and method name (for printing purposes)
def set_classifier(method):
    method_name = {
    "knn": "K-Nearest Neighbour Classification",
    "linear": "Linear Classification",
    "rf": "Random Forest Classification",
    "svc": "Support Vector Classification",
    "gnb": "Gaussian Naive Bayes",
    "ann": "Artifical Neural Network"
}

    if (method == "knn"):
        classifier = KNeighborsClassifier(n_neighbors=5)

    elif (method == "linear"):
        classifier = LogisticRegression(solver = 'lbfgs',max_iter=100000)

    elif (method == "svc"):
        classifier = SVC(C=1.0, gamma='auto')

    elif (method == "rf"):
        classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    elif (method == "gnb"):
        classifier = GaussianNB()
    
    elif (method == "ann"):
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    else:
        print("\nError: Invalid method name:" + method + "\n")
        parser.print_help()
        sys.exit(0)
    return classifier, method_name[method]


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("F1 Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, scoring = scoring, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve 1 and 2
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    # Plot learning curve 3 and 4
    axes[1].grid()
    axes[1].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[1].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[1].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[1].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[1].legend(loc="best")

    # Plot learning curve 5 and 6
    axes[2].grid()
    axes[2].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[2].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[2].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[2].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[2].legend(loc="best")
    # # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt

# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------
# define command line arguments
parser = argparse.ArgumentParser(description='Learning curves for classification')
parser.add_argument('--in_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file. The last colum of the file is the desired output.')
parser.add_argument('--method1', '-m1', action="store", dest='method1', default='linear', required=False, help='First method: knn, linear, rf, svc, gnb, ann')
parser.add_argument('--method2', '-m2', action="store", dest='method2', default='rf', required=False, help='Second method: knn, linear, rf, svc, gnb, ann')
parser.add_argument('--method3', '-m3', action="store", dest='method3', default='svc', required=False, help='First method: knn, linear, rf, svc, gnb, ann')
parser.add_argument('--method4', '-m4', action="store", dest='method4', default='knn', required=False, help='Second method: knn, linear, rf, svc, gnb, ann')
parser.add_argument('--method5', '-m5', action="store", dest='method5', default='gnb', required=False, help='First method: knn, linear, rf, svc, gnb, ann')
parser.add_argument('--method6', '-m6', action="store", dest='method6', default='ann', required=False, help='Second method: knn, linear, rf, svc, gnb, ann')
parser.add_argument('--kfold', '-k', action="store", dest='kfold', default=5, required=False, help='Number of folds for cross-validation')
parser.add_argument('--num_splits', '-n', action="store", dest='num_splits', default=10, required=False, help='Number of folds for cross-validation')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename    = args.in_file
method1     = args.method1
method2     = args.method2
method3     = args.method3
method4     = args.method4
method5     = args.method5
method6     = args.method6
kf          = int(args.kfold)
num_splits  = int(args.num_splits)


# set two classifiers based on user choices
classifier1, method_name1 = set_classifier(method1)
classifier2, method_name2 = set_classifier(method2)
classifier3, method_name3 = set_classifier(method3)
classifier4, method_name4 = set_classifier(method4)
classifier5, method_name5 = set_classifier(method5)
classifier6, method_name6 = set_classifier(method6)

# load the dataset; header is first row
df = pd.read_csv(filename, header=0)
# remove non-integer columns to plot
df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
# separate input and output variables
varray  = df_slice.values
nc      = len(varray[0,:])-1
X       = varray[:,1:18]
y       = varray[:,nc]

fig, axes = plt.subplots(3, 2, figsize=(10, 15))

#############
# First model
#############
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
scoring = 'f1'
title = r"Learning Curves " + method_name1
plot_learning_curve(classifier1, title, X, y, axes=axes[:, 0], cv=cv, n_jobs=4)

##############
# Second model
##############
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
title = r"Learning Curves " + method_name2
#estimator = RandomForestCLassifier()
plot_learning_curve(classifier2, title, X, y, axes=axes[:, 1], cv=cv, n_jobs=4)

##############
# Third model
##############
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
title = r"Learning Curves " + method_name3
#estimator = RandomForestCLassifier()
plot_learning_curve(classifier3, title, X, y, axes=axes[:, 0], cv=cv, n_jobs=4)

##############
# Fourth model
##############
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
title = r"Learning Curves " + method_name4
#estimator = RandomForestCLassifier()
#plot_learning_curve(classifier4, title, X, y, axes=axes[:, 1], cv=cv, n_jobs=4)

##############
# Fifth model
##############
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
title = r"Learning Curves " + method_name5
#estimator = RandomForestCLassifier()
#plot_learning_curve(classifier5, title, X, y, axes=axes[:, 0], cv=cv, n_jobs=4)

##############
# Sixth model
##############
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
title = r"Learning Curves " + method_name6
#estimator = RandomForestCLassifier()
#plot_learning_curve(classifier6, title, X, y, axes=axes[:, 1], cv=cv, n_jobs=4)

# save plots in file
plt.savefig(f'learningCurves/learning_curves_f1score_combined_CFS.png')
plt.show()
