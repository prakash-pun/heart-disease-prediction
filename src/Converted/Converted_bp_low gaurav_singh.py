import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"../data/train_data.csv")

data['bp_lo'].max()

data.head()

print(data["bp_lo"].describe())
print(data["bp_lo"].isnull().sum())
print(data["bp_lo"].dtypes)

m_bp_lo = data.loc[:, 'bp_lo'].mean()
m_round=round(m_bp_lo,-1)
print(m_round)

data['bp_lo'].fillna(value=m_round, inplace=True)
print('Updated Dataframe:')
print(data)
print(data["bp_lo"].isnull().sum())

data.to_csv("fill data bp_lo.csv")

print("Original DataFrame:")
print(data["bp_lo"])
print()

#  Min-Max Scaling
def min_max_scaling(df, column_name):
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    df[column_name] = (df[column_name] - min_value) / (max_value - min_value)

min_max_scaling(data,"bp_lo" )

print("DataFrame after Min-Max Scaling:")
print(data["bp_lo"])
print()

corr=data["bp_lo"]. corr(data["cardio"])
print(corr)

c=data["bp_lo"]. corr(data["bp_high"])
print(c)

plt.hist(data["bp_lo"])
plt.xlabel('Blood Pressure Low')
plt.ylabel('Frequency')
plt.title('')
plt.show()

correlation_matrix = data[['bp_lo', 'bp_high', 'cardio']].corr()
correlation_matrix

dataplot = sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.show()