# ===================== #
# Compute Variance Inflation Factors to check for multicollinearity #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run:   python3  RFE_multicol.py 
# Loops through training splits to generate a csv file for each split, containing only the features RFE selects.
# ================= #

# Import relevant libraries
import pandas as pd
import numpy as np
import argparse
import sys
import os
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


directory = './scaled_training_sets/'

# Get a list of all the CSV files in the directory
files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith('training')]


for filename in files:
    # load the dataset
    merged_df = pd.read_csv(filename)

    df_extra = merged_df[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home']]
    print(merged_df.columns)

   
    X=merged_df.values[:,:23]
    Y=merged_df.values[:,27].astype(int)


    # Feature selection
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X,Y)

    rfe = RFE(model, n_features_to_select = 17)
    fit = rfe.fit(X,Y)

    cols = list(merged_df.columns[:23])
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index

    # print("The selected features from RFE are: ", selected_features_rfe)
    
    # Use regex to get only the 'trainingyear1-year2' from the file name
    filename_re = re.search('\/(\w+-\w+)\.', filename)
    if filename_re:
        new_filename = filename_re.group(1)

    # df_rfe_corrmatrix = pd.concat([merged_df[selected_features_rfe].reset_index(drop=True)], axis=1)

    df_rfe_clean = pd.concat([merged_df[selected_features_rfe].reset_index(drop=True), df_extra], axis=1)
    df_rfe_clean.to_csv (f'./RFE_splits1/RFE_{new_filename}.csv', index = False)




    