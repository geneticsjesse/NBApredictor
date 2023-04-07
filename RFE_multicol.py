# ===================== #
# Compute Variance Inflation Factors to check for multicollinearity #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run:   python3  multicollinearityCheck.py 
# Loops through training splits to generate a csv file for each split, containing only the features RFE selects.
# ================= #

# Import relevant libraries
import pandas as pd
import numpy as np
import argparse
import sys
import os
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# define command line arguments
# parser = argparse.ArgumentParser(description='Variance Inflation Factor check for multicollinearity')
# parser.add_argument('--in_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file.')

# # handle user errors
# try:
#     args = parser.parse_args()
# except:
#     parser.print_help()
#     sys.exit(0)

# # save arguments in separate variables
# filename = args.in_file

directory = './scaled_training_sets/'

# Get a list of all the CSV files in the directory
files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
# print(csv_files)

for filename in files:
    # load the dataset
    merged_df = pd.read_csv(filename)

    df_extra = merged_df[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home']]
    print(merged_df.columns)

    # Now that we created a new cleaned dataset using CFS, we will perform RFE on the uncleaned dataset, to see if RFE and CFS select the same top 17 features
    X=merged_df.values[:,:23]
    Y=merged_df.values[:,27].astype(int)


    # Feature selection
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X,Y)

    rfe = RFE(model, n_features_to_select = 17)
    fit = rfe.fit(X,Y)

    cols = list(merged_df.columns[:23])
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index

    print("The selected features from RFE are: ", selected_features_rfe)
    
    # print ("The selected features from CFS are: ", df_merged_cfs_clean.columns[12:])
    
    # Use regex to get only the 'trainingyear1-year2' from the file name
    filename_re = re.search('\/(\w+-\w+)\.', filename)
    if filename_re:
        new_filename = filename_re.group(1)

    # df_rfe_corrmatrix = pd.concat([merged_df[selected_features_rfe].reset_index(drop=True)], axis=1)

    df_rfe_clean = pd.concat([merged_df[selected_features_rfe].reset_index(drop=True), df_extra], axis=1)
    df_rfe_clean.to_csv (f'./RFE_splits1/RFE_{new_filename}.csv', index = False)


    # compute the vif for all given features
    # def compute_vif(considered_features):
        
    #     X = merged_df[considered_features]
    #     # the calculation of variance inflation requires a constant
    #     X['intercept'] = 1
        
    #     # create dataframe to store vif values
    #     vif = pd.DataFrame()
    #     vif["Variable"] = X.columns
    #     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    #     vif = vif[vif['Variable']!='intercept']
    #     return vif

    # # features to consider removing
    # considered_features = ['FG.', 'percent_3pt', 'percent_2pt', 'percent_FT', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'MarginOfVictory', 'StrengthOfSchedule', 'SimpleRatngSystem', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', 'X3PAr', 'TS.']

    # # compute vif 
    # print ("Variance Inflation Factors for all predictors are as follows: \n", compute_vif(considered_features).sort_values('VIF', ascending=False))

    # print ("We see that there are some values >10, so we can remove high-VIF features and re-compute VIF.\n")
    # # compute vif values after removing a feature
    # considered_features.remove('NRtg')
    # print ("Removing NRtg and re-computing VIF...\n")
    # print(compute_vif(considered_features))

    # # Still, we have some high values of VIF, on we go.
    # considered_features.remove('PTS')
    # print ("Removing PTS and re-computing VIF...\n")
    # print(compute_vif(considered_features))

    # # Still, we have some high values of VIF, on we go.
    # print ("Removing SimpleRatngSystem  and re-computing VIF...\n")
    # considered_features.remove('SimpleRatngSystem')
    # print(compute_vif(considered_features))

    # # We can still see some VIF values >5, so we will remove TS. as it has a VIF of ~8.3
    # print ("Removing MarginOfVictory and re-computing VIF")
    # considered_features.remove('MarginOfVictory')
    # print(compute_vif(considered_features))

    # print ("Great, now we do not have any features with VIF values >5! We can move forward to looking at pairwise comparisons among features.\n")

    # # Now that all VIF values are <5, we can re-plot our correlation matrix and determine if we are satisfied that we are no longer including correlated variables
    # # First, we have to create a new data frame without the four columns we removed above.
    # df_cfs_clean= merged_df.drop(['ORtg', 'SimpleRatngSystem', 'MarginOfVictory', 'TS.'], axis=1)
    # print(df_cfs_clean.columns)
    # # Create a new dataframe to only look at our continuous predictor variables
    # df_continuous = df_cfs_clean.iloc[:, 0:19]
    # print(df_continuous.columns)
    # # Create a new dataframe of the non-continous variables that we need, but not for now.
    # df_extra = merged_df[['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd', 'wl_home']]
    # print(merged_df.columns)
    # # Now, all of our VIF values are <5, but there are still some large correlations, so we will filter out anything with a correlation value >0.5
    # #Correlation with output variable
    # cor = df_continuous.corr()
    # cor_target = abs(cor)
    # #Selecting highly correlated features
    # relevant_features = cor_target[cor_target>0.5]
    # print ("The features with a pearson correlation coefficient >0.5 are: \n", relevant_features)

    # # We can see that we have three pairwise comparisons with Pearson correlation values >0.5 (Pace-percent_2pt, X3PAr-percent_2pt, and Pace-X3PAr). We will remove X3PAr and percent_2pt and re-evaluate the correlation matrix.
    # df_continuous = df_continuous.drop(['X3PAr', 'percent_2pt'], axis=1)
    # cor = df_continuous.corr()
    # cor_target = abs(cor)
    # #Selecting highly correlated features
    # relevant_features = cor_target[cor_target>0.5]
    # print ("After removing X3Par and percent_2pt, the features with a pearson correlation coefficient >0.5 are: \n",relevant_features)

    # print ("Great, now that we have no features with a VIF value >5 or a Pearson's correlation coefficient >0.5, we can plot our correlation matrix to visualize the relationships between features.\n")

    # Perfect, we have no more pairwise variables with a pearson correlation coefficient >0.5. In addition, we have no variables with a VIF value over 5. We plot our final correlation matrix and output our new data frame.
    # set figure size
    # plt.figure(figsize=(18,15))

    # # Generate a mask to onlyshow the bottom triangle
    # mask = np.triu(np.ones_like(df_rfe_corrmatrix.corr(), dtype=bool))

    # # generate heatmap
    # sns.heatmap(df_rfe_corrmatrix.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
    # plt.title('Correlation Coefficient Of Predictors')
    # plt.savefig(f'collinearityPlots/{new_filename}_corrmatrix.png')
    # plt.show()

    # Combine df_continuous and df_extra after removing pearson correlations >0.5 and VIF values >5
    # df_merged_cfs_clean = pd.concat([df_continuous.reset_index(drop=True), df_extra], axis=1)
    # print(df_merged_cfs_clean)
    # Quickly move wl_home to the last column prior to export
    # df_slice = df_slice[['game_yearEnd', 'FG.', 'percent_3pt', 'percent_FT', 'DRB', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'StrengthOfSchedule', 'DRtg', 'NRtg', 'Pace', 'FTr', 'wl_home']]

    # Now that we have performed correlation-based feature selection, our new data frame is ready for downstream analyses.
    # df_merged_cfs_clean.to_csv('masterScript_testRun/merged_df_outliers_removed_CFS.csv', index = False)

    