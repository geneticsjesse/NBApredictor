import os


os.system("python3 dataCleaning.py -gamedata gamedata.csv -teamdata combinedTeamData.csv") 
os.system("python3 trainTestSplit.py -in merged_df_subset.csv")
os.system("python3 outlierDetect_Scaling.py")
os.system("python3 RFE.py")
os.system("python3 featureImportance.py -base ./scaled_training_sets/training2015-2021_outliers_removed_scaled.csv -rfe ./RFE_splits/RFE_training2015-2021_outliers_removed_scaled.csv -rfe9 ./RFE_splits/train2015_2021_RFEcommon.csv")
os.system("python3 -W ignore learningCurves.py")
os.system("python3 -W ignore stacking_LearningCurves.py")
os.system("python3 -W ignore stackingClassifier.py")