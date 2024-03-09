import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data_s = pd.read_csv(r"C:\Users\tanik\Documents\Data\train_data.csv")

data_s.head()

print(data_s["bp_high"].describe())
print(data_s["bp_high"].dtypes)

print(data_s["bp_high"].isnull().sum())

print("Original DataFrame:")
print(data_s["bp_high"])
print()

#  Min-Max Scaling
def min_max_scaling(df, name):
    min_value = df[name].min()
    max_value = df[name].max()
    df[name] = (df[name] - min_value) / (max_value - min_value)

min_max_scaling(data_s,"bp_high" )

print("DataFrame after Min-Max Scaling:")
print(data_s["bp_high"])
print()

co1=data_s["bp_high"]. corr(data_s["cardio"])
print(co1)
co2=data_s["bp_high"]. corr(data_s["bp_lo"])
print(co2)

plt.hist(data_s["bp_high"])
plt.xlabel('Blood Pressure High')
plt.ylabel('Frequency')
plt.title('')
plt.show()

correlation_matrix = data_s[['bp_high', 'bp_lo', 'cardio']].corr()
correlation_matrix

dataplot = sns.heatmap(correlation_matrix, cmap="BuPu", annot=True)
plt.show()