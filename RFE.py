# ===================== #
# Compute Recursive Feature Elimination
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run:   python3  RFE_multicol.py 
# This script loops through training splits to generate a csv file for each split, containing only the features RFE selects.
# ================= #

# Import relevant libraries
import pandas as pd
import os
import re
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

print ("\nBeginning RFE.py.\n")

# Set the directory where the scaled data are located
directory = './scaled_training_sets/'

# Make directory if does not exist
path = "RFE_splits"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

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

    rfe = RFE(model, n_features_to_select=17)
    fit = rfe.fit(X,Y)

    cols = list(merged_df.columns[:23])
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    
    # Use regex to get only the 'trainingyear1-year2' from the file name
    filename_re = re.search('\/(\w+-\w+)\.', filename)
    if filename_re:
        new_filename = filename_re.group(1)

    df_rfe_clean = pd.concat([merged_df[selected_features_rfe].reset_index(drop=True), df_extra], axis=1)
    df_rfe_clean.to_csv(f'./RFE_splits/RFE_{new_filename}.csv', index = False)

### Creating test set with same features as RFE_training2015-2021_outliers_removed_scaled.csv
RFE_training = pd.read_csv(f"./RFE_splits/RFE_training2015-2021_outliers_removed_scaled.csv")
scaled_test_set = pd.read_csv(f"./scaled_training_sets/test_set_outliers_removed_scaled.csv")

scaled_test_set = scaled_test_set.drop([col for col in scaled_test_set.columns if col not in RFE_training.columns], axis=1)
scaled_test_set.to_csv(f'./RFE_splits/test_RFE_all.csv', index=False)

# This code should loop over all files in the scaled_training_sets directory and only process the ones with the .csv file extension and ending with RFE.csv. It then loads each data frame from the CSV file and stores it in a dictionary using the file name as the key. Finally, it creates a list of column names for each data frame and prints them to the console.
directory = "RFE_splits"
file_extension = "RFE"

# # Create an empty list to store each dataframe
frames = []
# # Loop over all files in the directory
for file_name in os.listdir(directory):
    # Check if the file has the correct file extension
    if file_name.startswith(file_extension):
        # Load the data frame from the CSV file
        df = pd.read_csv(os.path.join(directory, file_name))
        frames.append(df)
# Create a list of the columns(features) that are shared among all datasets.
common_cols = list(set.intersection(*(set(df.columns) for df in frames)))


# # Print the common column names, or a message if there are no common column names
# if len(common_columns) > 0:
#     print(f"Common column names: {common_columns}")
# else:
#     print("No common column names")

# Read in train and test sets to create our RFE-common datasets
train2015_2021 = pd.read_csv('./RFE_splits/RFE_training2015-2021_outliers_removed_scaled.csv')
test = pd.read_csv('./scaled_training_sets/test_set_outliers_removed_scaled.csv') 

# Reset both train and test dataset to only contain the RFE_common columns
train2015_2021 = train2015_2021.loc[:, common_cols]
test = test.loc[:, common_cols]

df_extra_train = train2015_2021[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home']]
df_extra_test = test[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home']]
# print(df_extra_train)
# print(df_extra_test)

train2015_2021 = train2015_2021.drop(['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home'], axis=1)
test = test.drop(['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home'], axis=1)
# print(train2015_2021)
# print(test)
train2015_2021 = pd.concat([train2015_2021,df_extra_train], axis=1)
test = pd.concat([test,df_extra_test], axis=1)
#print(train2015_2021)
#print(test)


# Re-index columns to be in same order as the rest of the datasets
# train2015_2021 = train2015_2021.reindex(columns=['percent_3pt', 'percent_2pt', 'DRB', 'ORB', 'TRB', 'STL', 'DRtg', 'NRtg', 'TS.','team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd','wl_home'])
# test = test.reindex(columns=['percent_3pt', 'percent_2pt', 'DRB', 'ORB', 'TRB', 'STL', 'DRtg', 'NRtg', 'TS.','team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd','wl_home'])


# Create new csvs for downstream use
train2015_2021.to_csv('./RFE_splits/train2015_2021_RFEcommon.csv', index=False)
test.to_csv('./RFE_splits/test_RFEcommon.csv', index = False)    

print ("RFE.py has finished running, on to featureImportance.py.\n")