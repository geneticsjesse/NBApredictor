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

X,y = read_data()

# Scale the data to facilitate feature selection
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

#print (X)
print (X_scaled.columns)