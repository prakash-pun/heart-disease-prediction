"""alco_corr_cardio.py"""

import os
import pandas as pd
from matplotlib import pyplot as plt

current_dir = os.getcwd()
file_path = os.path.join("D:\\College\\Assignments\\Steps\\cardiovascular-disease-prediction\\data\\cleaned_data.xlsx")

df = pd.read_excel(file_path)

print(df.head(5))
print(df.info())

print(df.isnull().head(5))
print(df["alco"].isnull())
print(df["alco"].isna().all())
print(df["alco"].isin([0,1]).all())

df.plot(kind='scatter', x='active', y='bp_lo')
plt.gca().spines[['top', 'right',]].set_visible(False)