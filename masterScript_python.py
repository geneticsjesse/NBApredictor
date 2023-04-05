import os
gamedata = "gamedata.csv"
teamdata = "combinedTeamData.csv"
os.system("dataCleaning.py -gamedata gamedata -teamdata teamdata")
os.system("outlierDetection.py")