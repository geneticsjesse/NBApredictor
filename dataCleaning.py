# ===================== #
# Clean data prior to downstream analyses #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 15, 2023

# How to run:   python3  dataCleaning.py  -gamedata gamedata.csv -teamdata combinedTeamData.csv
# ================= #

# Import relevant libraries
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt 

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
teamdat_cleaned = teamdat[['Team', 'FG.', 'X3P.', 'X2P.', 'FT.', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', 'X3PAr', 'TS.', 'yearEnd.y']].rename(columns={'yearEnd.y': 'game_yearEnd','X3P.': 'percent_3pt','X2P.': 'percent_2pt','FT.': 'percent_FT', 'MOV': 'MarginOfVictory','SOS': 'StrengthOfSchedule', 'SRS': 'SimpleRatngSystem'})

# Left merging dataframes to create a master data frame
merged_df = pd.merge(gamedat_cleaned, teamdat_cleaned, on=['game_yearEnd', 'Team'], how='left')

# summarize data #
# -------------- #
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

# explore data visually #
# --------------------- #

# box and whisker plots
merged_df.iloc[:, 12:].plot(kind='box', subplots=True, layout=(4,6), sharex=False, sharey=False)
plt.show()

# histograms
merged_df.iloc[:, 12:].hist()
plt.show()

# Pie chart of wl_home
wl_home_count_list = merged_df["wl_home"].value_counts().tolist()
wl_home_list = merged_df["wl_home"].value_counts().keys().tolist()

sliceColors = ['#00AFBB', "#E7B800"]
plt.pie(wl_home_count_list, 
        labels = wl_home_list, 
        colors = sliceColors, 
        startangle=90,
        autopct='%.2f%%', 
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 1,
                      'antialiased': True})
plt.title("Win/Loss Home Distribution")
#plt.savefig(f"wl_home_piechart.png")
plt.show()

# The pie chart produced above shows us the home team wins ~57% of the time, meaning we will need to stratify our data when performing cross validation, to avoid an imbalanced training/testing split.

# Converting wl_home column to be binary (0,1) for Loss/Win respectively
merged_df['wl_home'] = merged_df['wl_home'].map({'W': 1, 'L': 0})

# Using the .isnull() method to find if there are any missing values in the merged dataset.
print("There are", merged_df.isnull().sum().sum(), "missing values in the merged dataset.") 

# Writing to csv
#merged_df.to_csv('merged_df.csv', index=False)