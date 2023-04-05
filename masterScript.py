# ===================== #
# Master Script - Wolf and Papp-Simon
# ===================== #

# ===================== #
# dataCleaning.py - Clean data prior to downstream analyses #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 15, 2023

# How to run:   python3  masterScript.py  -gamedata gamedata.csv -teamdata combinedTeamData.csv
# ================= #

# Import relevant libraries
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt 
import numpy as np
from pandas.api.types import is_numeric_dtype
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

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
print("There are", merged_df.isnull().sum().sum(), "missing values in the merged dataset. \n") 

# Writing to csv
merged_df.to_csv('masterScript_testRun/merged_df.csv', index=False)

# ===================== #
# outlierDetection.py #
# ===================== #

# select the columns to analyze
cols_to_analyze = merged_df.columns[12:]

# loop through the selected columns
for col in cols_to_analyze:
    print(f"Column {col}:")
    data = merged_df[col].values

    # calculate the inter-quartile range
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    #print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

    # calculate the outlier cutoff: k=1.5
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    # identify outliers
    data_outliers = [x for x in data if x < lower or x > upper]
    print('Number of identified outliers: %d' % len(data_outliers))
    #print('Outliers: ', data_outliers)

    # remove outliers
    data_outliers_removed = [x for x in data if x >= lower and x <= upper]
    print('Number of non-outlier observations: %d' % len(data_outliers_removed))
    # visualization
    # density=False would make counts
    # plt.hist(data_outliers_removed, density=True, bins=30, ec="blue")
    # plt.hist(data_outliers, density=True, bins=30, ec="red")
    # plt.ylabel('Probability')
    # plt.xlabel('Data')
    # plt.title({col})
    # plt.savefig(f'outlierPlots/outliers_iqr_prob_{col}.png')
    # plt.show()

    # plt.hist(data_outliers_removed, density=False, bins=30, ec="blue")
    # plt.hist(data_outliers, density=False, bins=30, ec="red")
    # plt.ylabel('Counts')
    # plt.xlabel('Data')
    # plt.title({col})
    # plt.savefig(f'outlierPlots/outliers_iqr_counts_{col}.png')
    # plt.show()
    
    # overwrite original data frame with non-outlier data
    #reset_index() method is called on the new non-outlier data to reset its index before assigning it to the dataframe column. The drop=True argument is used to drop the old index and replace it with a new one that starts from 0. This ensures that the new data has the same length as the dataframe index and can be assigned to the column without raising a ValueError.
    merged_df[col] = pd.Series(data_outliers_removed).reset_index(drop=True)

# Create a for loop to iterate over all columns in the the dataframe and replace NAs with mean values.
for col in merged_df.columns:
    if (is_numeric_dtype(merged_df[col])):
        merged_df[col] = merged_df[col].replace(np.NaN,merged_df[col].mean())

# Write to new file
merged_df.to_csv ('masterScript_testRun/merged_df_outliers_removed.csv', index = False)

# ===================== #
# Perform univariate (Correlation-based Feature Selection) and mulviariate (Recursive Feature Elimination)
# ===================== #
# compute the vif for all given features
def compute_vif(considered_features):
    
    X = merged_df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

# features to consider removing
considered_features = ['FG.', 'percent_3pt', 'percent_2pt', 'percent_FT', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'MarginOfVictory', 'StrengthOfSchedule', 'SimpleRatngSystem', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', 'X3PAr', 'TS.']

# compute vif 
print ("Variance Inflation Factors for all predictors are as follows: \n", compute_vif(considered_features).sort_values('VIF', ascending=False))

print ("We see that there are some values >10, so we can remove high-VIF features and re-compute VIF.\n")
# compute vif values after removing a feature
considered_features.remove('ORtg')
print ("Removing ORtg and re-computing VIF...\n")
print(compute_vif(considered_features))

# Still, we have some high values of VIF, on we go.
considered_features.remove('SimpleRatngSystem')
print ("Removing SimpleRatngSystem and re-computing VIF...\n")
print(compute_vif(considered_features))

# Still, we have some high values of VIF, on we go.
print ("Removing MarginOfVictory and re-computing VIF...\n")
considered_features.remove('MarginOfVictory')
print(compute_vif(considered_features))

# We can still see some VIF values >5, so we will remove TS. as it has a VIF of ~8.3
print ("Removing TS and re-computing VIF")
considered_features.remove('TS.')
print(compute_vif(considered_features))

print ("Great, now we do not have any features with VIF values >5! We can move forward to looking at pairwise comparisons among features.\n")

# Now that all VIF values are <5, we can re-plot our correlation matrix and determine if we are satisfied that we are no longer including correlated variables
# First, we have to create a new data frame without the four columns we removed above.
df_cfs_clean= merged_df.drop(['ORtg', 'SimpleRatngSystem', 'MarginOfVictory', 'TS.'], axis=1)

# Create a new dataframe to only look at our continuous predictor variables
df_continuous = df_cfs_clean.iloc[:, 12:]
# Create a new dataframe of the non-continous variables that we need, but not for now.
df_extra = df_cfs_clean[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home']]

# Now, all of our VIF values are <5, but there are still some large correlations, so we will filter out anything with a correlation value >0.5
#Correlation with output variable
cor = df_continuous.corr()
cor_target = abs(cor)
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print ("The features with a pearson correlation coefficient >0.5 are: \n", relevant_features)

# We can see that we have three pairwise comparisons with Pearson correlation values >0.5 (Pace-percent_2pt, X3PAr-percent_2pt, and Pace-X3PAr). We will remove X3PAr and percent_2pt and re-evaluate the correlation matrix.
df_continuous = df_continuous.drop(['X3PAr', 'percent_2pt'], axis=1)
cor = df_continuous.corr()
cor_target = abs(cor)
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print ("After removing X3Par and percent_2pt, the features with a pearson correlation coefficient >0.5 are: \n",relevant_features)

print ("Great, now that we have no features with a VIF value >5 or a Pearson's correlation coefficient >0.5, we can plot our correlation matrix to visualize the relationships between features.\n")

# Perfect, we have no more pairwise variables with a pearson correlation coefficient >0.5. In addition, we have no variables with a VIF value over 5. We plot our final correlation matrix and output our new data frame.
# set figure size
plt.figure(figsize=(18,15))

# Generate a mask to onlyshow the bottom triangle
mask = np.triu(np.ones_like(df_continuous.corr(), dtype=bool))

# generate heatmap
sns.heatmap(df_continuous.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Coefficient Of Predictors')
#plt.savefig(f'collinearityPlots/predictorCorrelationMatrix_reduced.png')
plt.show()

# Combine df_continuous and df_extra after removing pearson correlations >0.5 and VIF values >5
df_merged_cfs_clean = pd.concat([df_continuous.reset_index(drop=True), df_extra], axis=1)

# Quickly move wl_home to the last column prior to export
#df_slice = df_slice[['game_yearEnd', 'FG.', 'percent_3pt', 'percent_FT', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'StrengthOfSchedule', 'DRtg', 'NRtg', 'Pace', 'FTr', 'wl_home']]

# Now that we have performed correlation-based feature selection, our new data frame is ready for downstream analyses.
df_merged_cfs_clean.to_csv ('masterScript_testRun/merged_df_outliers_removed_CFS.csv', index = False)

# Now that we created a new cleaned dataset using CFS, we will perform RFE on the uncleaned dataset, to see if RFE and CFS select the same top 17 features
X=merged_df.values[:,12:]
Y=merged_df.values[:,7].astype(int)

# Prior to performing feature selection, we will scale the input features
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Feature selection
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_scaled,Y)
rfe = RFE(model, n_features_to_select = 17)
fit = rfe.fit(X,Y)

cols = list(merged_df.columns[12:])
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index

print("The selected features from RFE are: ", selected_features_rfe)

print ("The selected features from CFS are: ", df_merged_cfs_clean.columns[12:])

df_rfe_clean = pd.concat([merged_df[selected_features_rfe].reset_index(drop=True), df_extra], axis=1)
df_rfe_clean.to_csv ('masterScript_testRun/merged_df_outliers_removed_RFE.csv', index = False)