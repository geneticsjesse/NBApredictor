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
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import argparse
import sys

# define command line arguments
parser = argparse.ArgumentParser(description='Data cleaning')
parser.add_argument('--input_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
in_filename = args.in_file

# load the dataset
df = pd.read_csv(in_filename)

# x = 0
# print(df_CFS['game_yearEnd'].value_counts(2022))
# print(df_CFS.groupby('game_yearEnd').count(2022))
# count_2022 = len(df_CFS[df_CFS['game_yearEnd'] == 2022])
# print("Training test size corresponding to number of matches played in 2022:", str(count_2022)) # should be 1230 records for 2022.

# Checking variance of each column
# for col in df_CFS.columns:
#     if df_CFS[col].dtype == 'object':
#         df_CFS[col] = df_CFS[col].astype(int)
#     # if df_CFS[col].var() < 1:
#     #     df_CFS = df_CFS.drop(col, axis=1)
#     print(df_CFS[col].var())



### Function for time-series training-testing split, to ensure training is done on past season data to predict future seasons. 
# Takes in a dataframe, a specific model, and predictors. The start argument is set to 2, which means that we require at least 2 seasons in our training set to start making predictions (can be overwritten). The step argument refers to 
def backtest(df, model, predictors, start = 2, step = 1):
    # List of dataframes where each df contains the predictions for a single season
    all_predictions = []

    # Initialize a list of all seasons in our dataset
    seasons = sorted(df["game_yearEnd"].unique())

    # Loop through seasons, where we specify the requirement for at least 2 seasons to make predictions, up to the length of our 'seasons' list. The step parameter defines how we progress through the seasons we want to predict (so if step = 2, it will make predictions for 2 seasons at a time).
    for i in range(start, len(seasons), step):
        # First time going through loop -> i = 2 -> takes the 3rd element from the 'seasons' list, and will make predictions for that season. It will use data from all seasons prior to this one for the training data. Each time it goes through the loop, goes to next season (if step = 1).
        season = seasons[i] 

        train = df[df["game_yearEnd"] < season] # All data that comes before current season
        test = df[df["game_yearEnd"] == 2022] # Data used to generate predictions (current season)
        print(train, test)

seasons = sorted(df["game_yearEnd"].unique())
train_set_list = []
    # Loop through seasons, where we specify the requirement for at least 2 seasons to make predictions, up to the length of our 'seasons' list. The step parameter defines how we progress through the seasons we want to predict (so if step = 2, it will make predictions for 2 seasons at a time).
for i in range(2, len(seasons), 1):
        # First time going through loop -> i = 2 -> takes the 3rd element from the 'seasons' list, and will make predictions for that season. It will use data from all seasons prior to this one for the training data. Each time it goes through the loop, goes to next season (if step = 1).
    season = seasons[i] 

    train = df[df["game_yearEnd"] < season] # All data that comes before current season
    test = df[df["game_yearEnd"] == 2022] # Data used to generate predictions (current season)
    train_set_list.append(train) # List of training set dataframes
    

    # Write each training split to csv file
    for list in train_set_list:
        #print(list)
        list.to_csv(f"training2015-{season-1}.csv", index=False)
    
test.to_csv(f"testing2022.csv", index=False)


    #print(split_list)
        # model.fit(train[predictors], train["wl_home"])

        # preds = model.predict(test[predictors])
        # preds = pd.Series(preds, index=test.index) # Convert numpy array to pandas series

        # combined = pd.concat([test["wl_home"], preds], axis=1)
        # combined.columns = ["actual", "prediction"]
    # return pd.concat(all_predictions)






