import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../data/scaled_without_bin.csv")
ds2=data.copy()

print(ds2.tail())

print(ds2.info())

print(ds2.isnull().sum())

ds2 = ds2.drop(columns=["id","cardio"], axis=1)

correlation_matrix =ds2.corr(method='spearman').round(2)
correlation_matrix

fig, ax = plt.subplots(figsize=(25,25))
dataplot = sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.show()

#for positive values
threshold=0.15
relative_correl=correlation_matrix['cardio.1']
raw_cols=relative_correl[abs(relative_correl)>threshold].index.tolist()
raw_features=ds2[raw_cols]
raw_features.head()

new_correl=raw_features.corr(method='spearman')
sns.heatmap(new_correl, cmap="YlGnBu", annot=True)


for col in raw_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(ds2[col])
    plt.show()



