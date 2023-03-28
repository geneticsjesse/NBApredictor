# ===================== #
# Clean data prior to outlier analysis #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 15, 2023

# How to run:   python3  dataCleaning.py  -gamedata gamedata.csv -teamdata combinedTeamData.csv
# ================= #

# Import relevant libraries
import pandas as pd
import argparse
import sys

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

#gamedat = pd.read_csv(r"C:\Users\Admin\Documents\MBINF\ANSC6100\NBApredictor-1\gamedata.csv")
#teamdat = pd.read_csv(r"C:\Users\Admin\Documents\MBINF\ANSC6100\NBApredictor-1\combinedTeamData.csv")

# Renaming values containing 'LA Clippers' to 'Los Angeles Clippers' for consistency
gamedat['team_name_home'] = gamedat['team_name_home'].replace(to_replace="LA Clippers", value="Los Angeles Clippers")
gamedat['team_name_away'] = gamedat['team_name_away'].replace(to_replace="LA Clippers", value="Los Angeles Clippers")

gamedat_cleaned = gamedat[['season_id', 'team_id_home', 'team_abbreviation_home', 'team_name_home', 'game_id', 'game_date', 'matchup_home', 'wl_home', 'team_id_away', 'team_abbreviation_away', 'team_name_away', 'game_yearEnd']].rename(columns={'team_name_home': 'Team'})

# Renaming columns
teamdat_cleaned = teamdat[['Team', 'FG.', 'X3P.', 'X2P.', 'FT.', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', 'X3PAr', 'TS.', 'yearEnd.y']].rename(columns={'yearEnd.y': 'game_yearEnd','X3P.': 'percent_3pt','X2P.': 'percent_2pt','FT.': 'percent_FT', 'MOV': 'MarginOfVictory','SOS': 'StrengthOfSchedule', 'SRS': 'SimpleRatngSystem'})

# Left merging dataframes to create a master data frame
merged_df = pd.merge(gamedat_cleaned, teamdat_cleaned, on=['game_yearEnd', 'Team'], how='left')

# Converting wl_home column to be binary (0,1) for Loss/Win respectively
merged_df['wl_home'] = merged_df['wl_home'].map({'W': 1, 'L': 0})

# Adding new column that represents home/away status
merged_df['status'] = 1

# Writing to csv
merged_df.to_csv('merged_df_home.csv', index=False)

# Data pre-processing (missing values)

# Using the .isnull() method to find if there are any missing values in the merged dataset.
print(merged_df.isnull().sum().sum()) 