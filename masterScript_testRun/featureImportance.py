# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run: python3 .\featureImportance.py -base .\scaled_training_sets\training2015-2021_outliers_removed_scaled.csv -rfe .\RFE_splits1\RFE_training2015-2021_outliers_removed_scaled.csv -rfe9 .\RFE_splits\train2015_2021_RFEcommon.csv

# This script takes our baseline data (2015-2021 with no RFE performed), our RFE-selected data for 2015-2021, and our common RFE data (2015-2021 with only the 9 features that RFE identified as common between all training splits.)
# ================= #

# Import relevant libraries
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from matplotlib import pyplot
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description='Feature importance for the 3 datasets')
parser.add_argument('--base_file', '-base', action="store", dest='base_file', required=True, help='Name of csv input file.')
parser.add_argument('--RFE_file', '-rfe', action="store", dest='rfe_file', required=True, help='Name of csv input file.')
parser.add_argument('--RFE9_file', '-rfe9', action="store", dest='rfe9_file', required=True, help='Name of csv input file.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename_base = args.base_file
filename_rfe_all = args.rfe_file
filename_rfe_common = args.rfe9_file

# load the datasets
df_base = pd.read_csv(filename_base)
df_RFE_all = pd.read_csv(filename_rfe_all)
df_RFE_common = pd.read_csv(filename_rfe_common)

# print(df_base.columns)

df_base_features = df_base.drop(['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd'],axis=1)
df_RFE_common_features = df_RFE_common.drop(['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd'],axis=1)
df_RFE_all_features = df_RFE_all.drop(['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd'],axis=1)

df_list = [df_base_features, df_RFE_common_features]

# for df in df_list:
fig, axs = pyplot.subplots(1,2, figsize = (10, 7))

for df, ax in zip(df_list, axs.ravel()):
    X=df.values[:,]
    Y=df.values[:,-1:].astype(int)

    # Feature selectionn
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X,Y.ravel()) 
    RFE_model = RFE(estimator=model, n_features_to_select=len(X))

    # Create a data frame of importance and column names
    importances = pd.DataFrame(data={
    'Attribute': df.columns[0:],
    'Importance': model.coef_[0]
    })
    # Sort in descending order
    importances = importances.sort_values(by='Importance', ascending=False)
    importances = importances.iloc[1:]
    ax.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    ax.set_ylabel('Importance')
    ax.tick_params(axis='x', labelrotation=90)

    name =[x for x in globals() if globals()[x] is df][0]
    if name == 'df_base_features':
        ax.set_title('Feature Importance (baseline)')
    elif name == 'df_RFE_common_features':
        ax.set_title('Feature Importance (RFE common features)')

pyplot.savefig(f"./featureImportance/feature_Importance_base_RFE.png")


# Plotting the feature importance plot for all features selected by RFE for 2015-2021 training set separately.
fig, axs = pyplot.subplots(1,1, figsize = (10, 7))

X=df_RFE_all_features.values[:,]
Y=df_RFE_all_features.values[:,-1:].astype(int)

# Feature selectionn
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X,Y.ravel()) 
RFE_model = RFE(estimator=model, n_features_to_select=len(X))

# Create a data frame of importance and column names
importances = pd.DataFrame(data={
'Attribute': df_RFE_all_features.columns[0:],
'Importance': model.coef_[0]
})

# Sort in descending order
importances = importances.sort_values(by='Importance', ascending=False)
importances = importances.iloc[1:]
axs.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
axs.set_ylabel('Importance')
axs.tick_params(axis='x', labelrotation=90)

name =[x for x in globals() if globals()[x] is df_RFE_all_features][0]
if name == 'df_RFE_all_features':
    axs.set_title("Feature Importance (RFE all features)")

pyplot.savefig(f"./featureImportance/feature_Importance_base_RFE_all.png")