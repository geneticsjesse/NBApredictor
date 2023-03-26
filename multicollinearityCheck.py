# ===================== #
# Compute Variance Inflation Factors to check for multicollinearity #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run:   python3  multicollinearityCheck.py -in merged_df_outliers_removed.csv
# ================= #

# Import relevant libraries
import pandas as pd
import numpy as np
import argparse
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns

# define command line arguments
parser = argparse.ArgumentParser(description='Variance Inflation Factor check for multicollinearity')
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

# Slice the dataframe to only look at our continuous predictor variables
df_slice = df.iloc[:, 12:]

# set figure size
plt.figure(figsize=(18,15))

# Generate a mask to onlyshow the bottom triangle
mask = np.triu(np.ones_like(df_slice.corr(), dtype=bool))

# generate heatmap
sns.heatmap(df_slice.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Coefficient Of Predictors')
plt.savefig(f'collinearityPlots/predictorCorrelationMatrix.png')
#plt.show()

# compute the vif for all given features
def compute_vif(considered_features):
    
    X = df[considered_features]
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
#print (compute_vif(considered_features).sort_values('VIF', ascending=False))

# We see that there are some values >10, so we can remove and re-compute VIF.
# compute vif values after removing a feature
considered_features.remove('ORtg')
#print(compute_vif(considered_features))

# Still, we have some high values of VIF, on we go.
considered_features.remove('SimpleRatngSystem')
#print(compute_vif(considered_features))

# Still, we have some high values of VIF, on we go.
considered_features.remove('MarginOfVictory')
print(compute_vif(considered_features))

# Let's plot our correlation matrix again to see how it looks after removing ORtg, SimpleRatngSystem, and MarginOfVictory

# First, we have to create a new data frame without the three columns we removed above.
df_slice = df_slice.drop(['ORtg', 'SimpleRatngSystem', 'MarginOfVictory'], axis=1)

# set figure size
plt.figure(figsize=(18,15))

# Generate a mask to onlyshow the bottom triangle
mask = np.triu(np.ones_like(df_slice.corr(), dtype=bool))

# generate heatmap
sns.heatmap(df_slice.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Coefficient Of Predictors')
plt.savefig(f'collinearityPlots/predictorCorrelationMatrix_reduced.png')
#plt.show()