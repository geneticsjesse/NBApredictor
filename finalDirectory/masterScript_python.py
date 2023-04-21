import os


os.system("python dataCleaning.py -gamedata gamedata.csv -teamdata combinedTeamData.csv") 
os.system("python trainTestSplit.py -in merged_df_subset.csv")
os.system("python outlierDetect_Scaling.py")
os.system("python RFE.py")
os.system("python featureImportance.py -base ./scaled_training_sets/training2015-2021_outliers_removed_scaled.csv -rfe ./RFE_splits/RFE_training2015-2021_outliers_removed_scaled.csv -rfe9 ./RFE_splits/train2015_2021_RFEcommon.csv")
os.system("python learningCurves.py")
os.system("python stacking_LearningCurves.py")
os.system("python -W ignore stackingClassifier.py")