import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/scaled_dataset.csv')
data.head()

data.describe()

correlation_matrix = data.corr()
correlation_matrix

fig, ax = plt.subplots(figsize=(25,25))
dataplot = sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.show()

# correlation_matrix
correlation_with_cardio = correlation_matrix['cardio'].drop('cardio')

print('Correlation with Cardio:')
print(correlation_with_cardio)

#Bivariate analysis
sns.scatterplot(x='height', y='weight', data=data)
plt.title('Scatter Plot between Age and Weight')
plt.show()

sns.scatterplot(x='bp_high', y='bp_lo', data=data)
plt.title('Scatter Plot between BP high and BP low')
plt.show()

# Numerical features
numerical_features = ['age', 'height', 'weight', 'bp_high', 'bp_lo', 'bmi']
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='cardio', y=feature, data=data)
    plt.title(f'Box Plot of {feature} for Cardio Classes')
    plt.show()

for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=feature, y='cardio', data=data)
    plt.title(f'Scatter Plot of {feature} against Cardio')
    plt.show()

# Categorical features
categorical_features = [ 'cholesterol', 'gluc', 'diabetic']
for feature in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=feature, hue='cardio', data=data)
    plt.title(f'Count Plot of Cardio Classes by {feature}')
    plt.show()

for feature in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=feature, y='cardio', data=data)
    plt.title(f'Scatter Plot of {feature} against Cardio')
    plt.show()

binary_features = ['gender', 'smoke', 'alco', 'active','cholesterol_1','cholesterol_2','cholesterol_3','gluc_1','gluc_2','gluc_3','diabetic_1','diabetic_2','diabetic_3']

for feature in binary_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=feature, hue='cardio', data=data)
    plt.title(f'Stacked Bar Plot of Cardio Classes by {feature}')
    plt.show()

#Cholesterol
scaled_features = ['cholesterol', 'gluc', 'diabetic']

for feature in scaled_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=feature, y='cholesterol', data=data)
    plt.title(f'Scatter Plot of {feature} against Cardio')
    plt.show()