# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run: python3 featureSelection.py -in merged_df_outliers_removed_CFS.csv
# ================= #

#from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from matplotlib import pyplot
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description='Feature selection with Recursive Feature Elimination')
parser.add_argument('--CSF_file', '-csf', action="store", dest='csf_file', required=True, help='Name of csv input file.')
parser.add_argument('--RFE_file', '-rfe', action="store", dest='rfe_file', required=True, help='Name of csv input file.')
# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename_csf = args.csf_file
filename_rfe = args.rfe_file

# load the dataset
df_CSF = pd.read_csv(filename_csf)
df_RFE = pd.read_csv(filename_rfe)

df_list = [df_CSF, df_RFE]

# Separate input and output variables
# varray = df.values

# ncols = len(varray[0,:])-1
#X = varray[:,12:] # All continuous variables
#Y = varray[:,7] # Win/Loss
for df in df_list:
    X=df.values[:,0:17]
    Y=df.values[:,-1].astype(int)

    # Scale the data to facilitate feature selection
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Feature selection
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X,Y) ####### Here we can specify X or X_scaled
    # rfe = RFE(model, n_features_to_select = 5)
    # fit = rfe.fit(X,Y)

    #Perform permutation importance
    importance = model.coef_[0]
    x_axis = [x for x in range(len(importance))]

    pyplot.figure(figsize=(10,7))
    pyplot.bar(x_axis, importance)
    pyplot.xlabel('Features')
    pyplot.ylabel('Importance')
    pyplot.xticks(range(0,17))
    pyplot.savefig(f"featureImportance/feature_Importance_{0}.png".format(df))
    pyplot.show()

# feature_importance(df_RFE)
# feature_importance(df_CSF)

# for i, v in enumerate(importance):
#     print ('Feature: %0d, Score: %.5f' % (i,v))

# print("Num features: %d" % fit.n_features_)
# print("Selected features: %s" % fit.support_)
# print("Feature Ranking: %s" % fit.ranking_)

