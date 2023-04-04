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

tscv = TimeSeriesSplit(n_splits = 4)
rmse = []
for train_index, test_index in tscv.split(df_CFS):
    cv_train, cv_test = df_CFS.iloc[train_index], df_CFS.iloc[test_index]
    
    arma = sm.tsa.ARMA(cv_train, (2,2)).fit(disp=False)
    
    predictions = arma.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.values
    rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))
    
print("RMSE: {}".format(np.mean(rmse)))






