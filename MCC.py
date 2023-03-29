# ===================== #
# Calculate MCC for all models #
# ===================== #
# ===================== #
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     March 18, 2023

# How to run: python3 MCC.py -in merged_df_outliers_removed_CFS.csv | merged_df_outliers_removed_RFE.csv
# ================= #

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, RepeatedKFold, KFold
#from matplotlib import pyplot
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description='Fit each model and compute MCC')
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
X=df.iloc[:, 0:17]
Y=df.iloc[:,-1].astype(int)

# Scale the data to facilitate feature selection
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Split data into test and train two ways
# Using train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=13)

# Manually training on seasons 2015-2021 and testing on 2022
X_train_forward = scaler.transform(df[(df['game_yearEnd'] >= 2015) & (df['game_yearEnd'] <= 2021)].iloc[:, 0:17])
X_test_forward = scaler.transform(df[(df['game_yearEnd']== 2022)].iloc[:, 0:17])
Y_train_forward = df[(df['game_yearEnd'] >= 2015) & (df['game_yearEnd'] <= 2021)].iloc[:, -1]
Y_test_forward = df[(df['game_yearEnd']== 2022)].iloc[:, -1]

# Create function to evaluate MCC of a given model
def evaluate_model (X, Y, model):
    # Define cv method
    cvKF = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    # Define train test split
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)
    # Fit model
    model.fit (X, Y)
    # Create predictions
    Y_pred = cross_val_predict (model, X, Y, cv=cvKF)
    # Evaluate model
    scores = matthews_corrcoef (Y, Y_pred)
    return (scores)

# Perform logistic regression
#model = LogisticRegression(solver='lbfgs', max_iter=1000)
#model = KNeighborsClassifier(n_neighbors=5)
#model = SVC(C=1.0, gamma='auto')
#model = RandomForestClassifier(n_estimators=100, random_state=0)
#model = GaussianNB()
model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
results = evaluate_model(X_scaled, Y, model)

#print (f"The Matthews Correlation Coefficient from {model} was: ", results)



#logistic.fit(X_train_forward, Y_train_forward)

# Compute MCC
# Y_pred_forward = logistic.predict(X_test_forward)
# #print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic.score(X_test_forward, Y_test_forward)))

# confusion_matrix = confusion_matrix(y_test, y_pred)

# MCC_logisticRegression = matthews_corrcoef (Y_test_forward, Y_pred_forward)

# # Doing the same as above for knn
# neigh = KNeighborsClassifier(n_neighbors=5)