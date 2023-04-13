# ================= #
Revolutionizing NBA Predictions with Machine Learning: A Data-Driven Approach
Authors:Jesse Wolf & Thomas Papp-Simon
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
Getting Started
Prerequisites
To use this classifier, you will need to have the following installed:

Python (version 3.6 or higher)
Pandas
Scikit-learn
Matplotlib
Seaborn
Numpy

Clone this repository using the following command:
git clone https://github.com/geneticsjesse/NBApredictor

To use the classifier, run the masterScript_python.py script. The script will perform all the necessary steps to clean the data, split it into training and test sets, detect and remove outliers, scale the features, perform recursive feature selection, compute feature importance, generate learning curves, optimize hyperparameters, and test each model on unseen data.