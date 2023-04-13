# ========================================================================= #
# Training and Test data splitting 
#
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     April 4, 2023
#
# How to run:   python3 trainTestSplit.py -in merged_df.csv
# ========================================================================= #

import pandas as pd
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
        list.to_csv(f"./training_test_splits/training2015-{season-1}.csv", index=False)
    
test.to_csv(f"./training_test_splits/testing2022.csv", index=False)








