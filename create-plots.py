import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
from pprint import pprint 
import numpy as np

df = pd.read_csv("dataframes/20m-power.csv")
df["Average Power"] = df.apply(lambda row: np.mean([
        row["GPU 0"],
        row["GPU 1"],
        row["GPU 2"],
        row["GPU 3"],
        row["GPU 4"],
        row["GPU 5"],
        row["GPU 6"],
        row["GPU 7"],
    ]),
    axis=1
)
df = df.sort_values(by="timestamp")
print(df)

df_subsampled = df.iloc[::1000, :]
print(df_subsampled)

# sns.lineplot(x=df_subsampled.index, y="Average Power", data=df_subsampled)
sns.lineplot(x=df.index, y="Average Power", data=df)
plt.xlim(0, 50)
plt.ylim(0, 800)
plt.show()