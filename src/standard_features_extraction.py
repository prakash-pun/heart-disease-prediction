import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import vizualize as viz
from utils import get_data

# Data Import
df2 = get_data("scaled_standard_dataset.csv")
df2_copy = df2.copy(deep=True)

# Display the first few rows of the dataset
print(df2_copy.head())

# Display basic information about the dataset
print(df2_copy.info())

# Check for missing values
print(df2_copy.isnull().sum())

# Display descriptive statistics for the dataset
print(df2_copy.describe(include='all'))

# Visualize the correlation heatmap for the entire dataset
raw_correlation = df2_copy.corr(method='spearman').round(2)
viz.heatmap(raw_correlation)

# Feature Engineering with a new threshold
new_threshold = 0.15
relative_correlation = raw_correlation['cardio']
selected_cols = relative_correlation[abs(relative_correlation) > new_threshold].index.tolist()
selected_features = df2_copy[selected_cols]

# Visualize the correlation heatmap for selected features
new_correlation = selected_features.corr(method='spearman').round(2)
viz.heatmap(new_correlation)

# Plot histograms for selected features
for col in selected_cols:
    viz.histplot(df2_copy[col])

# Define a function to plot scatter plots for selected features
def scatter_plot(col):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=selected_features, x=col, y=selected_features.cardio)
    plt.show()

# Plot scatter plots for selected features
for cols in selected_cols:
    scatter_plot(cols)

# Save the selected features to a new CSV file
selected_features.to_csv("Standardscale_extraction_file.csv", index=False)