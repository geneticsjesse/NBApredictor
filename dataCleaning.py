import pandas as pd

gamedat = pd.read_csv(r"C:\Users\thoma\Documents\BINF-Sem2\ANSC6100\NBA_Machine_Learning\NBApredictor\gamedata.csv")
teamdat = pd.read_csv(r"C:\Users\thoma\Documents\BINF-Sem2\ANSC6100\NBA_Machine_Learning\NBApredictor\combinedTeamData.csv")

# Renaming values containing 'LA Clippers' to 'Los Angeles Clippers' for consistency
gamedat['team_name_home'] = gamedat['team_name_home'].replace(to_replace="LA Clippers", value="Los Angeles Clippers")
gamedat['team_name_away'] = gamedat['team_name_away'].replace(to_replace="LA Clippers", value="Los Angeles Clippers")

gamedat_cleaned = gamedat[['season_id', 'team_id_home', 'team_abbreviation_home', 'team_name_home', 'game_id', 'game_date', 'matchup_home', 'wl_home', 'team_id_away', 'team_abbreviation_away', 'team_name_away', 'game_yearEnd']].rename(columns={'team_name_home': 'Team'})

teamdat_cleaned = teamdat[['Team', 'FG.', 'X3P.', 'X2P.', 'FT.', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', 'X3PAr', 'TS.', 'yearEnd.y']].rename(columns={'yearEnd.y': 'game_yearEnd','X3P.': 'percent_3pt','X2P.': 'percent_2pt','FT.': 'percent_FT', 'MOV': 'MarginOfVictory','SOS': 'StrengthOfSchedule', 'SRS': 'SimpleRatngSystem'})


merged_df = pd.merge(gamedat_cleaned, teamdat_cleaned, on=['game_yearEnd', 'Team'], how='left')

merged_df.to_csv('NBApredictor/merged_df.csv', index=False)

# Data pre-processing (missing values)

# Using the .isnull() method to find if there are any missing values in the merged dataset.
print(merged_df.isnull().sum().sum()) 


