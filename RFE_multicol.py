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
    df_rfe_clean.to_csv (f'./RFE_splits/RFE_{new_filename}.csv', index = False)

#This code should loop over all files in the scaled_training_sets directory and only process the ones with the .csv file extension and ending with RFE.csv. It then loads each data frame from the CSV file and stores it in a dictionary using the file name as the key. Finally, it creates a list of column names for each data frame and prints them to the console.
directory = "RFE_splits1"
file_extension = ".csv"

# # Create an empty dictionary to store the data frames and their column names
data_frames = {}

# # Loop over all files in the directory
for file_name in os.listdir(directory):
    # Check if the file has the correct file extension
    if file_name.endswith(file_extension):
        # Load the data frame from the CSV file
        df = pd.read_csv(os.path.join(directory, file_name))
        # Store the data frame in the dictionary using the file name as the key
        data_frames[file_name] = df
        # Store the column names in a list for each data frame
        column_names = list(df.columns)
        # Print the column names for this data frame
       # print(f"Column names for {file_name}: {column_names} \n")

#This code creates a set of the column names for the first data frame in the dictionary, then loops over all the other data frames and updates the set of common column names with the intersection of the current data frame's columns and the existing set. Finally, it prints the resulting set of common column names.

# Create a list of the column names for the first data frame in the dictionary
common_columns = list(data_frames[list(data_frames.keys())[0]].columns)

# Loop over all other data frames in the dictionary
for key in list(data_frames.keys())[1:]:
    # Create a new list to store the common column names for this data frame
    current_common_columns = []
    # Loop over the column names in this data frame
    for col in data_frames[key].columns:
        # Check if this column name is in the list of common column names so far
        if col in common_columns:
            # If it is, add it to the current list of common column names
            current_common_columns.append(col)
    # Replace the list of common column names with the current list
    common_columns = current_common_columns

# Print the common column names, or a message if there are no common column names
if len(common_columns) > 0:
    print(f"Common column names: {common_columns}")
else:
    print("No common column names")

# Read in train and test sets to create our RFE-common datasets
train2015_2021 = pd.read_csv('RFE_splits1/RFE_training2015-2021.csv')
test = pd.read_csv('test_set_outliers_removed_scaled.csv') 

train2015_2021 = train2015_2021.loc[:, common_columns]
test = test.loc[:, common_columns]

train2015_2021.to_csv('./RFE_splits/train2015_2021_RFEcommon.csv', index=False)
test.to_csv('./RFE_splits/test_RFEcommon.csv', index = False)    