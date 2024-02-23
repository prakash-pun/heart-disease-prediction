import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_excel('cleaned_data.xlsx')

my_column = df[['active','cardio']]

sns.histplot(df['active'], bins = 20 , kde = True)
plt.title('Active Distribution')
plt.xlabel('active')
plt.ylabel('Frequency')
plt.show()

df.plot(kind='scatter', x='active', y='cardio')
plt.gca().spines[['top', 'right',]].set_visible(False)

correlation_matrix = df[['active','cardio']].corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation matrix of active and cardio')
plt.show()
