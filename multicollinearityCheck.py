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

# Slice the dataframe to only look at our continuous predictor variables and also retain wl_home (our output variable) and game_yearEnd, which we will use to split our train/tesitng data.
df_slice = df.iloc[:, 7:]
# Drop some columns we don't need
df_slice = df_slice.drop(['team_id_away', 'team_abbreviation_away', 'team_name_away'], axis=1)

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
print (compute_vif(considered_features).sort_values('VIF', ascending=False))

# We see that there are some values >10, so we can remove and re-compute VIF.
# compute vif values after removing a feature
considered_features.remove('ORtg')
print(compute_vif(considered_features))

# Still, we have some high values of VIF, on we go.
considered_features.remove('SimpleRatngSystem')
print(compute_vif(considered_features))

# Still, we have some high values of VIF, on we go.
considered_features.remove('MarginOfVictory')

# We can still see some high correlations and VIF values >5, so we will remove TS. as it has a VIF of ~8.3
considered_features.remove('TS.')
print(compute_vif(considered_features))

# Now that all VIF values are <5, we can re-plot our correlation matrix and determine if we are satisfied that we are no longer including correlated variables
# First, we have to create a new data frame without the three columns we removed above.
df_slice = df_slice.drop(['ORtg', 'SimpleRatngSystem', 'MarginOfVictory', 'TS.'], axis=1)

# Now, all of our VIF values are <5, but there are still some large correlations, so we will filter out anything with a correlation value >0.5
#Correlation with output variable
cor = df_slice.corr()
cor_target = abs(cor)
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print (relevant_features)

# We can see that we have three pairwise comparisons with Pearson correlation values >0.5 (Pace-percent_2pt, X3PAr-percent_2pt, and Pace-X3PAr). We will remove X3PAr and percent_2pt and re-evaluate the correlation matrix.
df_slice = df_slice.drop(['X3PAr', 'percent_2pt'], axis=1)
cor = df_slice.corr()
cor_target = abs(cor)
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
#print (relevant_features)

# Perfect, we have no more pairwise variables with a pearson correlation coefficient >0.5. In addition, we have no variables with a VIF value over 5. We plot our final correlation matrix and output our new data frame.
# set figure size
plt.figure(figsize=(18,15))

# Generate a mask to onlyshow the bottom triangle
mask = np.triu(np.ones_like(df_slice.iloc[:, 2:].corr(), dtype=bool))

# generate heatmap
sns.heatmap(df_slice.iloc[:, 2:].corr(), annot=True, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Coefficient Of Predictors')
#plt.savefig(f'collinearityPlots/predictorCorrelationMatrix_reduced.png')
plt.show()

# Now that we have performed correlation-based feature selection, our new data frame is ready for downstream analyses.
df_slice.to_csv ('merged_df_outliers_removed_CFS.csv', index = False)
