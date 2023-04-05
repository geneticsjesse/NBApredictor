# ===================== #
# Scaling X variables and outputting new csvs #
# ===================== #

# Source: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     April 4 2023

# How to run:   python3  scalingX.py  -in merged_df_outliers_removed_CFS.csv | merged_df_outliers_removed_RFE.csv
# ================= #

# Import libraries
import argparse
import sys
import os
import pandas as pd
from sklearn import preprocessing

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
# load the dataset
df = pd.read_csv(filename)
# Slice off non-numeric columns
#df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
# Set our x and y variables
X=df.values[:,0:17]
df_extra = df[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd']]
Y=df.values[:,-1].astype(int)

# Scale the data
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Create a new DataFrame with the scaled values and the original column names
scaled_df = pd.DataFrame(X_scaled)
# Concatenate our scaled_df with df_extra, which has our extra (categorical) data
scaled_df = pd.concat([scaled_df.reset_index(drop=True), df_extra], axis=1)

# Add the target column to the new DataFrame
scaled_df['wl_home'] = Y

print (scaled_df)

scaled_df.to_csv ('merged_df_outliers_removed_RFE_Xscaled.csv', index = False)