"""alco_corr_cardio.py"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

current_dir = os.getcwd()
file_path = os.path.join("D:\\College\\Assignments\\Steps\\cardiovascular-disease-prediction\\data\\cleaned_data.xlsx")

df = pd.read_excel(file_path)
df_cp = df.copy()

print(df_cp.head(5))
print(df_cp.info())

print(df_cp.isnull().head(5))
print(df_cp["alco"].isnull())
print(df_cp["alco"].isna().all())
print(df_cp["alco"].isin([0,1]).all())

df_cp.shape

correlation_matrix = df_cp[['alco', 'cardio']].corr()
correlation_matrix

sns.histplot(df_cp['alco'], bins=1, kde=True)
plt.title('Weight Distribution')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Alcohol and Cardio')
plt.show()