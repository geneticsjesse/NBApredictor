# ================= #
Revolutionizing NBA Predictions with Machine Learning: A Data-Driven Approach
Authors: Jesse Wolf & Thomas Papp-Simon
# ================= #

This machine learning classifier uses team statistics and head-to-head game data to predict future outcomes. The classifier uses a master file to perform the following tasks:

Data cleaning: The input data is cleaned to remove any missing or incorrect values.
Data splitting: The data is split into training and test sets to facilitate forward prediction.
Outlier detection: Outliers are detected and removed from the input data to ensure accurate prediction.
Scaling: All input features are scaled to ensure that they are on the same scale and to improve model performance.
Recursive Feature Selection: The most important features are selected using recursive feature selection to improve model performance.
Feature Importance: The importance of each feature is computed to identify the most important factors in the prediction.
Learning curves: Learning curves are generated for individual models (Random Forest, Logistic Regression, Naive Bayes, KNeighborsClassifier, MLPclassifier, and C-Support Vector Classification) as well as a stacking classifier to assess model performance.
Hyperparameter optimization: Hyperparameters for each of the individual models and the stacking classifier are optimized to improve model performance.
Testing: The performance of each model is tested on unseen data to assess the accuracy of the predictions.

Prerequisites
To use this classifier, you will need to have the following installed:

Python (version 3.6 or higher)
Pandas
Scikit-learn
Matplotlib
Numpy
Researchpy

For the sake of brevity, we have facilitated the creation of a merged_df_subset.csv (as part of the dataCleaning.py script) that will allow you to test the functionality of our pipeline. This .csv file is just a subset of our merged_df.csv that is output by the first script in the master script (dataCleaning.py). If you would like to run the full dataset, you can use the merged_df.csv as input to the second script in our master script (trainTestSplit.py). Using merged_df_subset.csv took approximately 15 minutes to complete on a computer with the following specifications: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz 3.19 GHz (4 cores; 8GB RAM). 

Note: If the classifier gets stuck at any point, you could consider lowering the number of splits and the number of repeats for the RepeatedStratitifedKFold cross-valdation function on lines 145, 198, and 229 of the stackingClassifier.py script, which should decrease the runtime of the program: 
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

To use the classifier, run 'python3 masterScript_python.py'. The script will read in the two .csv files included (gamedata.csv and combinedTeamData.csv) and perform all the necessary steps to clean the data, split it into training and test sets, detect and remove outliers, scale the features, perform recursive feature selection, compute feature importance, generate learning curves, optimize hyperparameters, and test each model on unseen data.

If you would like to clone this repository, use the following command:
git clone https://github.com/geneticsjesse/NBApredictor