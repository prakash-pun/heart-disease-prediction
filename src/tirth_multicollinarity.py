import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from vizualize import heat_map


df = pd.read_csv('scaled_dataset.csv')
df.head(7)

# correl=df.corr(method='pearson').round(2)
# plt.figure(figsize=(15,15))
# sns.heatmap(correl,annot=True)
# plt.show()

# correl=df.corr(method='spearman').round(2)
# plt.figure(figsize=(15,15))
# sns.heatmap(correl,annot=True)
# plt.show()

# correl=df.corr(method='kendall').round(2)
# plt.figure(figsize=(15,15))
# sns.heatmap(correl,annot=True)
# plt.show()

heat_map(df, "pearson")
heat_map(df, "spearman")
heat_map(df, "kendall")

