import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
from pprint import pprint 
import numpy as np

data = [
    {
        "Experiment": "<1B Dev.",
        "Power Usage": 18.86982805,
        "kL Water Consumption": 28.1160438,
        "CO2 Emissions (metric tons)": 6.26,
    },
    {
        "Experiment": "1B Dev.",
        "Power Usage": 109.2279239,
        "kL Water Consumption": 162.7496066,
        "CO2 Emissions (metric tons)": 36.26,
    },
    {
        "Experiment": "7B Dev.",
        "Power Usage": 195.5843993,
        "kL Water Consumption": 291.420755,
        "CO2 Emissions (metric tons)": 64.93,
    },
    {
        "Experiment": "MoE Dev.",
        "Power Usage": 18.56175271,
        "kL Water Consumption": 27.65701154,
        "CO2 Emissions (metric tons)": 6.16,
    },
    {
        "Experiment": "LM-20M-1.7T",
        "Power Usage": 0.7783589484,
        "kL Water Consumption": 1.159754833,
        "CO2 Emissions (metric tons)": 0.26,
    },
    {
        "Experiment": "LM-60M-1.7T",
        "Power Usage": 1.238497001,
        "kL Water Consumption": 1.845360531,
        "CO2 Emissions (metric tons)": 0.41,
    },
    {
        "Experiment": "LM-150M-1.7T",
        "Power Usage": 2.777493887,
        "kL Water Consumption": 4.138465892,
        "CO2 Emissions (metric tons)": 0.92,
    },
    {
        "Experiment": "LM-300M-1.7T",
        "Power Usage": 4.550209524,
        "kL Water Consumption": 6.77981219,
        "CO2 Emissions (metric tons)": 1.51,
    },
    {
        "Experiment": "LM-700M-1.7T",
        "Power Usage": 7.932200112,
        "kL Water Consumption": 11.81897817,
        "CO2 Emissions (metric tons)": 2.63,
    },
    {
        "Experiment": "LM-1B-3T",
        "Power Usage": 30.29130343,
        "kL Water Consumption": 45.13404212,
        "CO2 Emissions (metric tons)": 10.06,
    },
    {
        "Experiment": "LM-7B-2T",
        "Power Usage": 67.31812692,
        "kL Water Consumption": 100.3040091,
        "CO2 Emissions (metric tons)": 22.35,
    },
    {
        "Experiment": "LM-7B-3T",
        "Power Usage": 94.8885203,
        "kL Water Consumption": 141.383895,
        "CO2 Emissions (metric tons)": 31.5,
    },
    {
        "Experiment": "LM-7B-4T",
        "Power Usage": 156.7123195,
        "kL Water Consumption": 233.501356,
        "CO2 Emissions (metric tons)": 52.03,
    },
    {
        "Experiment": "LM-MoE-5T",
        "Power Usage": 54.1726267,
        "kL Water Consumption": 80.7172138,
        "CO2 Emissions (metric tons)": 17.99,
    },
]

df = pd.DataFrame.from_dict(data)
print(df)
df["Water Consumption (liters)"] = df.apply(lambda row: 1000 * row["kL Water Consumption"], axis=1)
# df["CO2 Emitted (grams)"] = df.apply(lambda row: 1000000 * row["CO2 Emissions (metric tons)"], axis=1)

fig, ax1 = plt.subplots()

color_palette = sns.color_palette("colorblind")

g1 = sns.scatterplot(x="Experiment", y="CO2 Emissions (metric tons)", s=100, ax=ax1, data=df, color=color_palette[0])
# g1 = sns.scatterplot(x="Experiment", y="CO2 Emitted (grams)", s=150, ax=ax1, data=df)
ax2 = ax1.twinx()
g2 = sns.scatterplot(x="Experiment", y="Water Consumption (liters)", s=100, ax=ax2, data=df, color=color_palette[0])
g1.set_yscale("log")
g2.set_yscale("log")
plt.setp(ax1.get_xticklabels(), rotation=315)
# for label in ax1.get_xticklabels():
    # label.set_rotation(315)
    # label.set_horizontalalignment('left')  # Adjust alignment to 'right'

# # Define tick positions (you can adjust this range based on your data)
# yticks = [10**i for i in range(-1, 6)]  # From 10^-1 to 10^3, adjust range as needed

# # Set ticks and labels for primary y-axis
# ax1.set_yticks(yticks)
# ax1.set_yticklabels([f'$10^{i}$' for i in range(-1, 6)], fontsize=12)

# # Set ticks and labels for secondary y-axis
# ax2.set_yticks(yticks)
# ax2.set_yticklabels([f'$10^{i}$' for i in range(-1, 6)], fontsize=12)

# Increase fontsize for x and y labels
ax1.set_xlabel(ax1.get_xlabel(), fontsize=14)
ax1.set_ylabel(ax1.get_ylabel(), fontsize=14)
label2 = ax2.set_ylabel(ax2.get_ylabel(), fontsize=14)

# Increase fontsize for tick labels
ax1.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Rotate the secondary y-axis label by 180 degrees
label2.set_rotation(270)

# Adjust position of the rotated label to avoid overlap or clipping
label2.set_va('bottom')  # Adjust vertical alignment
label2.set_ha('center')  # Adjust horizontal alignment
label2.set_position((1.1, 0.5))  # Shift label to avoid overlap with ticks

ax1.axhline(y=.8887, color=color_palette[2], linestyle='-', linewidth=2, label='100 gallons of gas')
ax1.axhline(y=43, color=color_palette[2], linestyle='-', linewidth=2, label='100 barrels of oil')
ax1.text(0.01, 1.087, '100 gallons of gas', color=color_palette[2], va='center', ha='left', transform=ax1.get_yaxis_transform())
ax1.text(0.01, 50, '100 barrels of oil', color=color_palette[2], va='center', ha='left', transform=ax1.get_yaxis_transform())

ax2.axhline(y=113297, color=color_palette[3], linestyle='-', linewidth=2, label='Yearly Water Usage (1 person, U.S.)')
ax2.text(0.01, 95000, 'Yearly Water Usage (1 person, U.S.)', color=color_palette[3], va='center', ha='left', transform=ax2.get_yaxis_transform())
ax2.axhline(y=9442, color=color_palette[3], linestyle='-', linewidth=2, label='Monthly Water Usage (1 person, U.S.)')
ax2.text(0.01, 11000, 'Monthly Water Usage (1 person, U.S.)', color=color_palette[3], va='center', ha='left', transform=ax2.get_yaxis_transform())

ax1.axvline(x=4 - 0.5, linestyle='--', color='gray', alpha=0.5)

# ax1.text(0.01, 70, 'tanker trucks\' worth of gasoline', color='green', va='center', ha='left', transform=ax1.get_yaxis_transform())

# plt.xlim(1000, 2000)
# plt.ylim(0, 800)



plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('plots/fig-1-test-6.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.clf()