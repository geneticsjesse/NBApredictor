# ===================== #
# Compare RFE selected features for all training sets #
# ===================== #

# Source: 
# Author:   Jesse Wolf, jwolf@uoguelph.ca | Thomas Papp-Simon, tpappsim@uoguelph.ca
# Date:     April 4 2023

# How to run:   python3  compareTrainingRFE.py
import os
import pandas as pd

# 2015 - 2021 training set
# training_2015_2021 = pd.read_csv('scaled_training_sets/training2015-2021.csv_outliers_removed_scaled.csv_RFE.csv')
# train_2015_2021_cols = training_2015_2021.columns

#This code should loop over all files in the scaled_training_sets directory and only process the ones with the .csv file extension and ending with RFE.csv. It then loads each data frame from the CSV file and stores it in a dictionary using the file name as the key. Finally, it creates a list of column names for each data frame and prints them to the console.
directory = "scaled_training_sets"
file_extension = "RFE.csv"

# Create an empty dictionary to store the data frames and their column names
data_frames = {}

# Loop over all files in the directory
for file_name in os.listdir(directory):
    # Check if the file has the correct file extension
    if file_name.endswith(file_extension):
        # Load the data frame from the CSV file
        df = pd.read_csv(os.path.join(directory, file_name))
        # Store the data frame in the dictionary using the file name as the key
        data_frames[file_name] = df
        # Store the column names in a list for each data frame
        column_names = list(df.columns)
        # Print the column names for this data frame
        #print(f"Column names for {file_name}: {column_names} \n")

#This code creates a set of the column names for the first data frame in the dictionary, then loops over all the other data frames and updates the set of common column names with the intersection of the current data frame's columns and the existing set. Finally, it prints the resulting set of common column names.
# Create a set of the column names for the first data frame in the dictionary
# common_columns = set(data_frames[list(data_frames.keys())[0]].columns)

# # Loop over all other data frames in the dictionary
# for key in list(data_frames.keys())[1:]:
#     # Update the set of common column names with the intersection of the current data frame's columns and the existing set
#     common_columns = common_columns.intersection(set(data_frames[key].columns))

# # Print the common column names, or a message if there are no common column names
# if len(common_columns) > 0:
#     print(f"Common column names: {common_columns}")
# else:
#     print("No common column names")
# Create a list of the column names for the first data frame in the dictionary
common_columns = list(data_frames[list(data_frames.keys())[0]].columns)

# Loop over all other data frames in the dictionary
for key in list(data_frames.keys())[1:]:
    # Create a new list to store the common column names for this data frame
    current_common_columns = []
    # Loop over the column names in this data frame
    for col in data_frames[key].columns:
        # Check if this column name is in the list of common column names so far
        if col in common_columns:
            # If it is, add it to the current list of common column names
            current_common_columns.append(col)
    # Replace the list of common column names with the current list
    common_columns = current_common_columns

# Print the common column names, or a message if there are no common column names
if len(common_columns) > 0:
    print(f"Common column names: {common_columns}")
else:
    print("No common column names")

import seaborn as sns
import matplotlib.pyplot as plt

# Create a list of the column names for each data frame
column_lists = [list(data_frames[key].columns) for key in data_frames]

# Compute the overlap between each pair of column lists
overlap_matrix = [[len(set(column_lists[i]).intersection(set(column_lists[j]))) for j in range(len(column_lists))] for i in range(len(column_lists))]

# Create a heatmap of the overlap matrix
sns.heatmap(overlap_matrix, cmap="YlGnBu", annot=True, fmt="d")
plt.xticks(range(len(data_frames)), list(data_frames.keys()), rotation=90)
plt.yticks(range(len(data_frames)), list(data_frames.keys()), rotation=0)
plt.xlabel("Data Frame")
plt.ylabel("Data Frame")
plt.title("Overlap between Common Columns")
plt.show()
