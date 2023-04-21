# ===================== #
# Clean data prior to downstream analyses #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 15, 2023

# How to run:   python  dataCleaning.py  -gamedata gamedata.csv -teamdata combinedTeamData.csv
# This script takes in raw game and team data, cleans, and merges them into a single dataframe. It also performs some initial data exploration and generates plots.
# ================= #

# Import relevant libraries
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt 
import os
from scipy.stats import shapiro

# Make directory if does not exist
path = "dataExploration"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

print ("\nBeginning dataCleaning.py.\n")

# define command line arguments
parser = argparse.ArgumentParser(description='Data cleaning')
parser.add_argument('--game_file', '-gamedata', action="store", dest='gameData_in_file', required=True, help='Name of csv input file for game data.')
parser.add_argument('--team_file', '-teamdata', action="store", dest='teamData_in_file', required=True, help='Name of csv input file for team data.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
game_filename = args.gameData_in_file
team_filename = args.teamData_in_file

# load the dataset
gamedat = pd.read_csv(game_filename)
teamdat = pd.read_csv (team_filename)

# Renaming values containing 'LA Clippers' to 'Los Angeles Clippers' for consistency
gamedat['team_name_home'] = gamedat['team_name_home'].replace(to_replace="LA Clippers", value="Los Angeles Clippers")
gamedat['team_name_away'] = gamedat['team_name_away'].replace(to_replace="LA Clippers", value="Los Angeles Clippers")

gamedat_cleaned = gamedat[['season_id', 'team_id_home', 'team_abbreviation_home', 'team_name_home', 'game_id', 'game_date', 'matchup_home', 'wl_home', 'team_id_away', 'team_abbreviation_away', 'team_name_away', 'game_yearEnd']].rename(columns={'team_name_home': 'Team'})

# Renaming columns
teamdat_cleaned = teamdat[['Team', 'FG.', 'X3P.', 'X2P.', 'FT.', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', 'X3PAr', 'TS.', 'yearEnd.y']].rename(columns={'yearEnd.y': 'game_yearEnd','X3P.': 'percent_3pt','X2P.': 'percent_2pt','FT.': 'percent_FT', 'MOV': 'MarginOfVictory'})#,'SOS': 'StrengthOfSchedule', 'SRS': 'SimpleRatngSystem'})

# Left merging dataframes to create a master data frame
merged_df = pd.merge(gamedat_cleaned, teamdat_cleaned, on=['game_yearEnd', 'Team'], how='left')

# summarize data
print('Data summarization')
print('------------------')
# shape
print('\nDataset size: ', merged_df.shape)

# head
print('\nFirst 10 lines of data:\n', merged_df.head(10))

# descriptions
print('\nSummary stats of data:\n', merged_df.describe())

# class distribution
print('\nClass distribution:\n', merged_df.groupby('team_abbreviation_home').size())

# explore data visually

# Pie chart of wl_home
wl_home_count_list = merged_df["wl_home"].value_counts().tolist()
wl_home_list = merged_df["wl_home"].value_counts().keys().tolist()

sliceColors = ['#00AFBB', "#E7B800"]
plt.pie(wl_home_count_list, 
        labels = wl_home_list, 
        colors = sliceColors, 
        startangle=90,
        autopct='%.2f%%', 
        textprops={'fontsize': 16},
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 1,
                      'antialiased': True})
plt.title("Win/Loss Home Distribution")
plt.tight_layout()
plt.savefig(f"./dataExplorationPlots/wl_home_piechart.png")
#plt.show()

# histograms
merged_df.iloc[:, 12:].hist()
plt.tight_layout()
#plt.show()
plt.savefig(f"./dataExplorationPlots/featuresHistogram.png")

# The pie chart produced above shows us the home team wins ~57% of the time, meaning we will need to stratify our data when performing cross validation, to avoid an imbalanced training/testing split.

# Converting wl_home column to be binary (0,1) for Loss/Win respectively
merged_df['wl_home'] = merged_df['wl_home'].map({'W': 1, 'L': 0})

# Using the .isnull() method to find if there are any missing values in the merged dataset.
print("There are", merged_df.isnull().sum().sum(), "missing values in the merged dataset.") 

##### Writing to csv
merged_df.to_csv('merged_df.csv', index=False)

##### Creating subsampled version of merged_df and writing it to csv for submission purposes

#group the dataframe by year
grouped = merged_df.groupby('game_yearEnd')

# initialize an empty list to store the subsets
subsets = []

# loop through each group and create a subset with 10% of the rows
for year, group in grouped:
    n = int(len(group) * 0.1)
    subset = group.sample(n=n)
    subsets.append(subset)

# concatenate the subsets into a single dataframe
subset_df = pd.concat(subsets)
subset_df.to_csv("merged_df_subset.csv", index=False)




#### Normality test
num_subsamples = 10
subsample_size = 200

normal_col_list = []
notnormal_col_list = []
for col in merged_df.columns[12:]:
    normal_count = 0
    print(f"Column: {col}")
    for i in range(num_subsamples):
        subsample = merged_df[col].sample(subsample_size)
        stat, p = shapiro(subsample)
        alpha = 0.05
        if p > alpha:
            # print(f"Subsample {i+1}: Normal (p={p:.3f})")
            normal_count += 1
        # else:
        #     print(f"Subsample {i+1}: Not normal (p={p:.3f})")
    if normal_count >= 6:
        print(f"{col} IS NORMAL!")
        normal_col_list.append(col)
    else:
        print(f"{col} IS NOT NORMAL!")
        notnormal_col_list.append(col)
print(f"List of normal columns:", normal_col_list)
print(f"List of not normal columns:", notnormal_col_list)

print ("dataCleaning.py has finished running, on to trainTestSplit.py\n")