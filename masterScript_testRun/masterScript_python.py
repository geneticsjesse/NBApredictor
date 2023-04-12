import os
#gamedata = "gamedata.csv"
#teamdata = "combinedTeamData.csv"
os.system("python3 dataCleaning.py -gamedata gamedata.csv -teamdata combinedTeamData.csv")
os.system("python3 trainTestSplit.py")