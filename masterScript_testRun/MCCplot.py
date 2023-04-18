#MCCplot

import pandas as pd
import matplotlib.pyplot as plt

# Create a dataframe with the data
dat = pd.read_csv('MCC_comparisons.csv', header = 0)
                  
# Define custom color palette
color_palette = [plt.cm.tab10(i) for i in range(len(dat["Model"].unique()))]

# Define dictionary to map dataset values to numerical values
dataset_dict = {"Baseline": 0, "RFE_common": 1, "RFE_all": 2}

# Create bar plot
plt.figure(figsize=(8, 6))
for i, model in enumerate(sorted(dat["Model"].unique())):
    subset = dat[dat["Model"] == model]
    x_values = [dataset_dict[d] + i*(0.8/len(dat["Model"].unique())) for d in subset["Dataset"]]
    y_values = subset["MCC"]
    plt.bar(x=x_values,
            height=y_values,
            width=0.8/len(dat["Model"].unique()),
            color=color_palette[i],
            edgecolor="black",
            alpha=0.8,
            align="center",
            label=model)

# Add title and axis labels
plt.title("Comparison of MCC scores on testing data")
plt.xlabel("Dataset")
plt.ylabel("Matthew's Correlation Coefficient")

# Customize axis tick labels
plt.xticks(ticks=[0, 1, 2], labels=["Baseline", "RFE_common", "RFE_all"], fontsize=12, ha="center")
plt.yticks(fontsize=12)
plt.gca().spines["left"].set_linewidth(1)
plt.gca().spines["bottom"].set_linewidth(1)

# Adjust position of legend
plt.legend(loc="upper center", ncol=len(dat["Model"].unique()), bbox_to_anchor=(0.5, 1.15))


# Customize plot background and gridlines
plt.gca().set_facecolor("white")
plt.grid(True, axis="y", linestyle="-", color="lightgrey", alpha=0.5)

# Save plot
plt.savefig('MCC_plot_py.png')