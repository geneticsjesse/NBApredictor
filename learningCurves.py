# ========================================================================= #
# Learning curves for classification (comparison between six methods)
# reference and inspiration
# source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 26, 2023
#
# How to run:   python3 learning_curves.py 
# This script plots learning curves to evaluate training and test performance of 6 models.
# ========================================================================= #

# Import relevant libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import LearningCurveDisplay, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print ("\nBeginning learningCurves.py. This one might take few minutes.\n")

# Make directory if does not exist
path = "learningCurves"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

# load the dataset; header is first row
df_base = pd.read_csv('./scaled_training_sets/training2015-2021_outliers_removed_scaled.csv', header=0)
df_rfe_common = pd.read_csv('./RFE_splits/train2015_2021_RFEcommon.csv', header=0)
df_rfe_all = pd.read_csv('./RFE_splits/RFE_training2015-2021_outliers_removed_scaled.csv', header=0)

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

    # Generate our initial models
    svc= SVC(C=1.0, gamma='auto')
    gnb =  GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=5)
    linear = LogisticRegression(max_iter=100000)
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    ann =  MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)

    # Create a list of models to iterate over
    model_list = [gnb, knn, ann, linear, rf, svc]
    # Set up our 3x2 grid for learning curve plots
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), squeeze=False)
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
    # Create a for loop to create a plot for each model and the corresponding learning curve
    for ax_idx, estimator in zip(ax.ravel(), model_list):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax_idx, scoring="matthews_corrcoef")
        handles, label = ax_idx.get_legend_handles_labels()
        ax_idx.legend(handles[:2], ["Training Score", "Test Score"])
        ax_idx.set_title(f"Learning Curve for {estimator.__class__.__name__}")
        
    # Get a list of dataframe names in the global environment
    name =[x for x in globals() if globals()[x] is df][0]

    # Adjust the subplot positions
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig(f'learningCurves/{name}_LearningCurves.png')    

print ("learningCurves.py has finished running, on to stacking_LearningCurves.py.\n")