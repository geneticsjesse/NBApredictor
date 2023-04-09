# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run: python3 featureimportance.py 
# ================= #

#from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from matplotlib import pyplot
import pandas as pd
import argparse
import sys

# parser = argparse.ArgumentParser(description='Feature selection with Recursive Feature Elimination')
# parser.add_argument('--CFS_file', '-cfs', action="store", dest='cfs_file', required=True, help='Name of csv input file.')
# parser.add_argument('--RFE_file', '-rfe', action="store", dest='rfe_file', required=True, help='Name of csv input file.')
# # handle user errors
# try:
#     args = parser.parse_args()
# except:
#     parser.print_help()
#     sys.exit(0)

# # save arguments in separate variables
# filename_cfs = args.cfs_file
# filename_rfe = args.rfe_file

# load the dataset
df_base = pd.read_csv('./scaled_training_sets/training2015-2021.csv_outliers_removed_scaled.csv')
df_RFE = pd.read_csv('training2015-2021.csv_outliers_removed_scaled_RFECOPY_JW.csv')

print(df_base.columns)

df_base_features = df_base.drop(['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd'],axis=1)
df_RFE_features = df_RFE.drop(['team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'game_yearEnd'],axis=1)

print(df_base_features.columns)
print(df_RFE_features.columns)
df_list = [df_base_features, df_RFE_features]

# Separate input and output variables
# varray = df.values

# ncols = len(varray[0,:])-1
#X = varray[:,12:] # All continuous variables
#Y = varray[:,7] # Win/Loss

# for df in df_list:
fig, axs = pyplot.subplots(1,2, figsize = (10, 7))

for df, ax in zip(df_list, axs.ravel()):
    X=df.values[:,]
    Y=df.values[:,-1:].astype(int)

    # Feature selectionn
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X,Y.ravel()) 
    RFE_model = RFE(estimator=model, n_features_to_select=len(X))
    # rfe = RFE(model, n_features_to_select = 5)
    # fit = rfe.fit(X,Y)

    # Create a data frame of importance and column names
    importances = pd.DataFrame(data={
    'Attribute': df.columns[0:24],
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
    else:
        ax.set_title('Feature Importance (RFE)')
    
    # ax.xticks(range(0,17), rotation = 'vertical')
    # pyplot.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    # # pyplot.xlabel('Features')
    # pyplot.ylabel('Importance')
    # pyplot.xticks(range(0,17), rotation = 'vertical')

pyplot.savefig(f"featureImportance/feature_Importance_base_RFE.png")
pyplot.show()



# print("Num features: %d" % fit.n_features_)
# print("Selected features: %s" % fit.support_)
# print("Feature Ranking: %s" % fit.ranking_)

