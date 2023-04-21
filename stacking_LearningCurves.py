# ========================================================================= #
# Learning curves for classification (stacking classifier only)
# reference and inspiration
# source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 26, 2023
#
# How to run:   python3 
# This script generates learning curves for our stacking classifier
# ========================================================================= #

# Import relevant libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

print ("\nBeginning stacking_LearningCurves.py. This one might take a few minutes\n")

# load the dataset; header is first row
df_base = pd.read_csv('./scaled_training_sets/training2015-2021_outliers_removed_scaled.csv', header=0)
df_rfe_common = pd.read_csv('./RFE_splits/train2015_2021_RFEcommon.csv', header=0)
df_rfe_all = pd.read_csv('./RFE_splits/RFE_training2015-2021_outliers_removed_scaled.csv', header=0)

# Create stacking function
def get_stacking():
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

# Create a list of our three dataframes to iterate over
df_list = [df_base, df_rfe_common, df_rfe_all]

for df in df_list:
    # remove non-integer columns to plot
    df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
    # separate input and output variables
    varray  = df_slice.values
    nc      = len(varray[0,:])-1
    X       = varray[:,0:nc]
    y       = varray[:,nc]
     # Generate our stacking model
    stacking = get_stacking()

    # Setting up our learning curve plot
    fig, ax = plt.subplots(1,1, figsize=(10, 7), squeeze=False)
    # Create a dictionary of common parameters for LearningCurveDisplay
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": RepeatedStratifiedKFold(n_splits=3, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Matthew's Correlation Coefficient",
    }
    # Generate our learning curve plot
    fig, ax = plt.subplots(1,1, figsize = (10, 7))
    LearningCurveDisplay.from_estimator(stacking, **common_params, ax=ax, scoring="matthews_corrcoef")
    ax.legend(["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve for Stacking Classifier")
    # Get a list of dataframe names in the global environment    
    name =[x for x in globals() if globals()[x] is df][0]
    plt.savefig(f'learningCurves/{name}_StackingClassifer_LearningCurves.png')

print ("stacking_LearningCurves.py has finished running, on to stackingClassifer.py\n")