# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run: python3 featureSelection.py -in merged_df_outliers_removed_CFS.csv
# ================= #

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from matplotlib import pyplot
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description='Feature selection with Recursive Feature Elimination')
parser.add_argument('--in_file', '-in', action="store", dest='in_file', required=True, help='Name of csv input file.')

# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# save arguments in separate variables
filename = args.in_file

# load the dataset
df = pd.read_csv(filename)

# Separate input and output variables
varray = df.values

# ncols = len(varray[0,:])-1
#X = varray[:,12:] # All continuous variables
#Y = varray[:,7] # Win/Loss
X=df.values[:,0:17]
Y=df.values[:,-1].astype(int)

# Scale the data to facilitate feature selection
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Feature selection
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_scaled,Y) ####### Here we can specify X or X_scaled
rfe = RFE(model, n_features_to_select = 17)
fit = rfe.fit(X,Y)

# Create a data frame of importance and column names
importances = pd.DataFrame(data={
    'Attribute': df.columns[0:17],
    'Importance': model.coef_[0]
})
# Sort in descending order
importances = importances.sort_values(by='Importance', ascending=False)

# Plot
pyplot.figure(figsize=(8,6))
pyplot.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
#pyplot.xlabel('Features')
pyplot.ylabel('Importance')
pyplot.xticks(range(0,17), rotation = 'vertical')
pyplot.title ('Feature Importance - CFS - X scaled', size = 20)
pyplot.savefig(f'featureImportance/feature_Importance_CFS_xScaled.png')
pyplot.show()

# for i, v in enumerate(importances):
#     print ('Feature: %0d, Score: %.5f' % (i,v))
