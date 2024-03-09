import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C://Users//cleaned_data.csv")

df.head()

df.isnull().sum()

df.shape

df.dropna(inplace=True)

df.isnull().sum()

# Summary statistics
df.describe()

sns.histplot(df['weight'], bins=20, kde=True)
plt.title('Weight Distribution')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

# Convert height from cm to meters (as BMI formula requires height in meters)
df['height_m'] = df['height'] / 100

# Calculate BMI
df['BMI'] = df['weight'] / (df['height_m'] ** 2)

# Round BMI to the nearest whole number
df['BMI'] = df['BMI'].round()

# Print the updated dataframe with BMI column
print(df[['weight', 'height', 'BMI']])

correlation_matrix = df[['weight', 'height', 'BMI','cardio']].corr()

correlation_matrix

# Plot correlation matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Weight, Height, BMI and Cardio')
plt.show()

df.head()

df.to_csv('data_update.csv', index=False)