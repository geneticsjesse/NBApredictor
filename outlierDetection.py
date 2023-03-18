# ===================== #
# Identify outliers using the IQR method #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 15, 2023

# How to run:   python3  outlierDetection.py  -in merged_df.csv
# ================= #

# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# define command line arguments
parser = argparse.ArgumentParser(description='Outlier Detection')
parser.add_argument('--in_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename = args.in_file

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
    cut_off = iqr * 1.5
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

df.to_csv ('merged_df_outliers_removed.csv', index = False)