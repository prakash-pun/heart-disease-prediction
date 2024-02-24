#all library import
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/cleaned_data.csv')
df.head()

df.info()

df.isnull().head(5)

df.isnull().sum()

# Alcohol intake | Subjective Feature | alco | binary (0, 1)
df.describe()

len(df)

df["alco"].isnull().all()

df['alco'].isin([0, 1]).all()

df.dropna(subset=['alco'], inplace=True) # Removing rows with missing 'alco' values

len(df)

alco_counts = df['alco'].value_counts()
alco_counts

alco_counts[0]

alco_counts.values

fig, ax = plt.subplots()

bar_labels = ['No', 'Yes']
bar_colors = ['tab:blue', 'tab:green']

bar_container = ax.bar(bar_labels, alco_counts.values, label=bar_labels, color=bar_colors)

ax.set_ylabel('Data Count')
ax.set_xlabel('Alcohol (0: No, 1: Yes)')
ax.set_title('Alcohol Consumption')

ax.bar_label(bar_container)
#plt.savefig('alco_bar.png')
