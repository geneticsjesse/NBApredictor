# ========================================================================= #
# Learning curves for classification (comparison between two methods)
# reference and inspiration
# source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 26, 2023
#
# How to run:   python3 learning_curves.py -in merged_df_outliers_removed_CFS.csv  -m1 knn  -m2 linear -n 10
# ========================================================================= #
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# load the dataset; header is first row
df_base = pd.read_csv('./scaled_training_sets/training2015-2021.csv_outliers_removed_scaled.csv', header=0)
df_rfe = pd.read_csv('training2015-2021.csv_outliers_removed_scaled_RFECOPY_JW.csv', header=0)
df_rfe_all = pd.read_csv('./RFE_splits1/RFE_training2015-2021.csv', header=0)

df_list = [df_base, df_rfe, df_rfe_all]

for df in df_list:
    # remove non-integer columns to plot
    df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
    # separate input and output variables
    varray  = df_slice.values
    nc      = len(varray[0,:])-1
    X       = varray[:,1:nc]
    y       = varray[:,nc]

    svc= SVC(C=1.0, gamma='auto')
    gnb =  GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=5)
    linear = LogisticRegression(max_iter=100000)
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    ann =  MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)

    model_list = [gnb, knn, ann, linear, rf, svc]

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), squeeze=False)

    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": RepeatedStratifiedKFold(n_splits=3, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "MCC",
    }

    # for ax_idx, estimator in enumerate(model_list):
    #     print(ax_idx)
    #     LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx, ax_idx+1])
    #     handles, label = ax[ax_idx, ax_idx+1].get_legend_handles_labels()
    #     ax[ax_idx, ax_idx+1].legend(handles[:2], ["Training Score", "Test Score"])
    #     ax[ax_idx, ax_idx+1].set_title(f"Learning Curve for {estimator.__class__.__name__}")

    for ax_idx, estimator in zip(ax.ravel(), model_list):
        print(ax_idx)
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax_idx, scoring="matthews_corrcoef")
        handles, label = ax_idx.get_legend_handles_labels()
        ax_idx.legend(handles[:2], ["Training Score", "Test Score"])
        ax_idx.set_title(f"Learning Curve for {estimator.__class__.__name__}")
        

    name =[x for x in globals() if globals()[x] is df][0]
    # if name == df_base:
    #     plt.suptitle("Learning Curves (Baseline)")
    # elif name == df_rfe:
    #     plt.suptitle("Learning Curves (RFE 9 Common Features)")
    # elif name == df_rfe_all:
    #     plt.suptitle("Learning Curves (RFE 2015-2021)")

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig(f'learningCurves/{name}_LearningCurves.png')    
    plt.show()

