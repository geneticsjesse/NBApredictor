# ========================================================================= #
# Training and Test data splitting 
#
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     April 4, 2023
#
# How to run:   python3 trainTestSplit.py -cfs merged_df_outliers_removed_CFS.csv
# ========================================================================= #

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.api as sm
import pandas as pd
import argparse
import sys

# define command line arguments
parser = argparse.ArgumentParser(description='Data cleaning')
parser.add_argument('--CFS_file', '-cfs', action="store", dest='cfs_file', required=True, help='Name of csv input file.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
cfs_filename = args.cfs_file

# load the dataset
df_CFS = pd.read_csv(cfs_filename)

x = 0
# print(df_CFS['game_yearEnd'].value_counts(2022))
# print(df_CFS.groupby('game_yearEnd').count(2022))
count_2022 = len(df_CFS[df_CFS['game_yearEnd'] == 2022])
print(str(count_2022)) # should be 1230 records for 2022.

# Checking variance of each column
for col in df_CFS.columns:
    if df_CFS[col].dtype == 'object':
        df_CFS[col] = df_CFS[col].astype(int)
    # if df_CFS[col].var() < 1:
    #     df_CFS = df_CFS.drop(col, axis=1)
    print(df_CFS[col].var())

# tscv = TimeSeriesSplit(n_splits = 4, test_size=count_2022) # make the max test size 1230
# rmse = []
# for train_index, test_index in tscv.split(df_CFS):
#     cv_train, cv_test = df_CFS.iloc[train_index], df_CFS.iloc[test_index]
    
#     arma = sm.tsa.ARIMA(cv_train, (2,2)).fit(disp=False)
    
#     predictions = arma.predict(cv_test.index.values[0], cv_test.index.values[-1])
#     true_values = cv_test.values
#     rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))
    
# print("RMSE: {}".format(np.mean(rmse)))






