import os
#gamedata = "gamedata.csv"
#teamdata = "combinedTeamData.csv"
# os.system("python3 dataCleaning.py -gamedata gamedata.csv -teamdata combinedTeamData.csv")
# os.system("python3 trainTestSplit.py -in merged_df.csv")
# os.system("python3 outlierDetect_Scaling.py")
# os.system("python3 RFE_multicol.py")
#os.system("python3 featureImportance.py -base scaled_training_sets/ -rfe RFE_splits/ -rfe9 ")
#os.system("python3 learningCurves.py")
#os.system("python3 stackingClassifier_2.py")

os.system("python dataCleaning.py -gamedata gamedata.csv -teamdata combinedTeamData.csv") 
os.system("python trainTestSplit.py -in merged_df.csv")
os.system("python outlierDetect_Scaling.py")
os.system("python RFE_multicol.py")
# Add feature importance
os.system("python learningCurves.py")
os.system("python stackingClassifier_2.py")