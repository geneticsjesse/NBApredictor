# ===================== #
# Identify outliers using the IQR method and scale our input features #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 15, 2023

# How to run:   python3  outlierDetect_Scaling.py
# This script reads in all training and test datasets, identifies and removes outliers, and scales the input features
# ================= #

# Import relevant librariess
import pandas as pd
import numpy as np
import os
import re
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype

print ("\nBeginning outlierDetect_Scaling.py.\n")

# Set the directory where the training and test data are located
directory = './training_test_splits/'

# Get a list of all the CSV files in the directory
files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith('training')]

# Make directory if does not exist
path = "scaled_training_sets"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

for filename in files:
# load the dataset
    df = pd.read_csv(filename)

    # select the columns to analyze
    cols_to_analyze = df.columns[12:]

    # loop through the selected columns
    for col in cols_to_analyze:

        data = df[col].values

        # calculate the inter-quartile range
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        iqr = q75 - q25
       
        # calculate the outlier cutoff: k=2.5
        cut_off = iqr * 2.5
        lower, upper = q25 - cut_off, q75 + cut_off

        # identify outliers
        data_outliers = [x for x in data if x < lower or x > upper]
       

        # remove outliers
        data_outliers_removed = [x for x in data if x >= lower and x <= upper]
        
        df[col] = pd.Series(data_outliers_removed).reset_index(drop=True)

    # Create a for loop to iterate over all columns in the the dataframe and replace NAs with mean values.
    for col in df.columns:
        if (is_numeric_dtype(df[col])):
            df[col] = df[col].replace(np.NaN,df[col].mean())

    # Write to new file
    # df.to_csv ('masterScript_testRun/df_outliers_removed.csv', index = False)

#########################

# ===================== #
# Scaling X variables and outputting new csvs #
# ===================== #

# Training set
# Set our x and y variables
    X=df.values[:,12:]
    # Create a list of column names to add to our new scaled dataframe
    X_colnames= df.columns[12:]
    # Create a data frame with our non-numerical data
    df_extra = df[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd']]
    Y=df.values[:,7].astype(int)

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Create a new DataFrame with the scaled values and the original column names
    scaled_df = pd.DataFrame(X_scaled, columns=X_colnames)
    # Concatenate our scaled_df with df_extra, which has our extra (categorical) data
    scaled_df = pd.concat([scaled_df.reset_index(drop=True), df_extra], axis=1)

    # Add the target column to the new DataFrame
    scaled_df['wl_home'] = Y

    # Use regex to get only the 'trainingyear1-year2' from the file name
    filename_re = re.search('\/(\w+-\w+)\.', filename)
    if filename_re:
        new_filename = filename_re.group(1)
    
    scaled_df.to_csv (f'./scaled_training_sets/{new_filename}_outliers_removed_scaled.csv', index = False)


# Testing set
# Set our x and y variables

test_set = pd.read_csv('./training_test_splits/testing2022.csv')
# Scale the data using the same scaler as was used on the training data, to avoid data leakage
X_test=test_set.values[:,12:]
Y_test=test_set.values[:,7].astype(int)
test_scaled = scaler.transform(X_test)

# Create a new DataFrame with the scaled values and the original column names
scaled_df_test = pd.DataFrame(test_scaled, columns=X_colnames)

df_extra_test = test_set[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd']]
# Concatenate our scaled_df with df_extra, which has our extra (categorical) data
scaled_df_test = pd.concat([scaled_df_test.reset_index(drop=True), df_extra_test], axis=1)

# Add the target column to the new DataFrame
scaled_df_test['wl_home'] = Y_test

scaled_df_test.to_csv (f'./scaled_training_sets/test_set_outliers_removed_scaled.csv', index = False)

print ("outlierDetect_Scaling.py has finished running, on to RFE.py\n")