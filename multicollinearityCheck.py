# ===================== #
# Compute Variance Inflation Factors to check for multicollinearity #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run:   python3  multicollinearityCheck.py -in merged_df_outliers_removed.csv
# ================= #

# Import relevant libraries
import pandas as pd
import argparse
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

#create DataFrame to hold VIF values
vif_df = pd.DataFrame()
vif_df['variable'] = df.columns[12:]

# Slice the dataframe to only look at our continuous predictor variables
df_slice = df.iloc[:, 12:]

#calculate VIF for each predictor variable 
vif_df['VIF'] = [variance_inflation_factor(df_slice.values, i) for i in range(df_slice.shape[1])]

#view VIF for each predictor variable 
print(vif_df)