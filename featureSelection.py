# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run: python3 featureSelection.py -in merged_df_outliers_removed.csv
# ================= #

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
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
X = varray[:,13:] # All continuous variables
Y = varray[:,8] # Win/Loss
# print(Y)

# Feature selection
model = LogisticRegression(solver='lbfgs', max_iter=1)
rfe = RFE(model, n_features_to_select = 10)
fit = rfe.fit(X,Y.astype(float))
print("Num features: %d" % fit.n_features_)
print("Selected features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

