# ===================== #
# Merge home and away datasets #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 28, 2023

# How to run:   python3  dataCleaning.py  -homedata merged_df_home.csv -awaydata merged_df_away.csv
# ================= #

# Import relevant libraries
import pandas as pd
import argparse
import sys

# define command line arguments
parser = argparse.ArgumentParser(description='Data cleaning')
parser.add_argument('--home_data', '-homedata', action="store", dest='homedata_in_file', required=True, help='Name of csv input file for home data.')
parser.add_argument('--away_data', '-awaydata', action="store", dest='awaydata_in_file', required=True, help='Name of csv input file for away data.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
home_filename = args.homedata_in_file
away_filename = args.awaydata_in_file

# load the dataset
homedat = pd.read_csv(home_filename)
awaydat = pd.read_csv (away_filename)

# Left merging dataframes to create a master data frame
merged_df_homeaway = pd.concat([homedat.reset_index(drop=True), awaydat], axis=0)
#merged_df_homeaway = pd.merge(homedat, awaydat, on=['game_id'], how='left')

merged_df_homeaway.to_csv('merged_df_homeaway.csv', index=False)