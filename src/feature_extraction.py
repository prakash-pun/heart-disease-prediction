import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import vizualize as viz
from utils import get_data

# Data Import

# scaled_standard_dataset.csv
def feature_extract(data_frame):
    df = data_frame.copy(deep=True)

    raw_correl = df.corr(method='spearman').round(2)
    viz.heatmap(raw_correl)

    threshold = 0.11
    relative_correl = raw_correl['cardio']
    raw_cols = relative_correl[abs(relative_correl) > threshold].index.tolist()
    raw_features = df[raw_cols]
    raw_features.head()

    new_correl = raw_features.corr(method='spearman').round(2)
    viz.heatmap(new_correl)

    for col in raw_cols:
        viz.histplot(df[col])

    def scatter(col):
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=raw_features, x=col, y=raw_features.cardio)
        plt.show()

    for cols in raw_cols:
        scatter(cols)

    raw_features.head()

    raw_features.to_csv("data/feature_extracted_raw.csv")

    return raw_features


data = pd.read_csv("../data/scaled_without_bin.csv")


def feature_extract_without_binary(data_frame):
    ds2 = data_frame.copy()

    ds2 = ds2.drop(columns=["id", "cardio"], axis=1)

    correlation_matrix = ds2.corr(method='spearman').round(2)
    correlation_matrix

    # for positive values
    threshold = 0.15
    relative_correl = correlation_matrix['cardio.1']
    raw_cols = relative_correl[abs(relative_correl) > threshold].index.tolist()
    raw_features = ds2[raw_cols]
    raw_features.head()

    new_correl = raw_features.corr(method='spearman')
    sns.heatmap(new_correl, cmap="YlGnBu", annot=True)

    for col in raw_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(ds2[col])
        plt.show()


feature_extract_without_binary(data)


def feature_extract_dataset():
    # Data Import
    df2 = get_data("scaled_standard_dataset.csv")
    df2_copy = df2.copy(deep=True)

    # Visualize the correlation heatmap for the entire dataset
    raw_correlation = df2_copy.corr(method='spearman').round(2)
    viz.heatmap(raw_correlation)

    # Feature Engineering with a new threshold
    new_threshold = 0.15
    relative_correlation = raw_correlation['cardio']
    selected_cols = relative_correlation[abs(
        relative_correlation) > new_threshold].index.tolist()
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
        sns.scatterplot(data=selected_features, x=col,
                        y=selected_features.cardio)
        plt.show()

    # Plot scatter plots for selected features
    for cols in selected_cols:
        scatter_plot(cols)

    # Save the selected features to a new CSV file
    selected_features.to_csv(
        "data/Standardscale_extraction_file.csv", index=False)

    return selected_features
