# ===================== #
# Identify outliers using the IQR method #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 15, 2023

# How to run:   python3  outlierDetection.py  -in df.csv
# ================= #

# Import relevant librariess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype

# # define command line arguments
# parser = argparse.ArgumentParser(description='Outlier Detection')
# parser.add_argument('--in_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file.')

# # handle user errors
# try:
#     args = parser.parse_args()
# except:
#     parser.print_help()
#     sys.exit(0)

# save arguments in separate variables
# filename = args.in_file
directory = './training_sets/'

# Get a list of all the CSV files in the directory
files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
# print(csv_files)

for filename in files:
# load the dataset
    df = pd.read_csv(filename)

    # select the columns to analyze
    cols_to_analyze = df.columns[12:]

    # loop through the selected columns
    for col in cols_to_analyze:
        print(f"Column {col}:")
        data = df[col].values

        # calculate the inter-quartile range
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        iqr = q75 - q25
        #print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

        # calculate the outlier cutoff: k=1.5
        cut_off = iqr * 2.5
        lower, upper = q25 - cut_off, q75 + cut_off

        # identify outliers
        data_outliers = [x for x in data if x < lower or x > upper]
        print('Number of identified outliers: %d' % len(data_outliers))
        #print('Outliers: ', data_outliers)

        # remove outliers
        data_outliers_removed = [x for x in data if x >= lower and x <= upper]
        print('Number of non-outlier observations: %d' % len(data_outliers_removed))
        # visualization
        # density=False would make counts
        # plt.hist(data_outliers_removed, density=True, bins=30, ec="blue")
        # plt.hist(data_outliers, density=True, bins=30, ec="red")
        # plt.ylabel('Probability')
        # plt.xlabel('Data')
        # plt.title({col})
        # plt.savefig(f'outlierPlots/outliers_iqr_prob_{col}.png')
        # plt.show()

        # plt.hist(data_outliers_removed, density=False, bins=30, ec="blue")
        # plt.hist(data_outliers, density=False, bins=30, ec="red")
        # plt.ylabel('Counts')
        # plt.xlabel('Data')
        # plt.title({col})
        # plt.savefig(f'outlierPlots/outliers_iqr_counts_{col}.png')
        # plt.show()
        
        # overwrite original data frame with non-outlier data
        #reset_index() method is called on the new non-outlier data to reset its index before assigning it to the dataframe column. The drop=True argument is used to drop the old index and replace it with a new one that starts from 0. This ensures that the new data has the same length as the dataframe index and can be assigned to the column without raising a ValueError.
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



#parser.add_argument('--method', '-m', action="store", dest='method', default='linear', required=False, help='Method: adaboost, ridgec, dtc, gbc, knn, linear, linsvc, mlp, rf, sgd, svc')
#parser.add_argument('--kfold', '-k', action="store", dest='kfold', default=5, required=False, help='Number of folds for cross-validation')
#parser.add_argument('--num_splits', '-n', action="store", dest='num_splits', default=3, required=False, help='Number of folds for cross-validation')

# # handle user errors
# try:
#     args = parser.parse_args()
# except:
#     parser.print_help()
#     sys.exit(0)

# # save arguments in separate variables
# filename    = args.in_file
# # load the dataset
# df = pd.read_csv(filename)
# Slice off non-numeric columns
#df_slice = df.drop(['game_date', 'team_abbreviation_home', 'team_abbreviation_away'], axis=1)
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

    print (scaled_df)
    scaled_df.to_csv (f'{filename}_outliers_removed_scaled.csv', index = False)

test_set = pd.read_csv('testing2022.csv')
test_scaled = scaler.transform(X)

# Create a new DataFrame with the scaled values and the original column names
scaled_df_test = pd.DataFrame(test_scaled, columns=X_colnames)
# Concatenate our scaled_df with df_extra, which has our extra (categorical) data
scaled_df_test = pd.concat([scaled_df_test.reset_index(drop=True), df_extra], axis=1)

# Add the target column to the new DataFrame
scaled_df_test['wl_home'] = Y

print (scaled_df_test)
scaled_df_test.to_csv (f'./scaled_training_sets/test_set_outliers_removed_scaled.csv', index = False)


#scaled_df.to_csv ('df_outliers_removed_RFE_Xscaled.csv', index = False)