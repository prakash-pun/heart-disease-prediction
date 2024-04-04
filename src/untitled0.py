# from visualize import Visualizer
from data_loading import DataInitializer
import matplotlib.pyplot as plt
import numpy as np
#Modules
reader = DataInitializer()
# plotter = Visualizer()

X_train, X_test, y_train, y_test = reader.split_data()

def get_counts(col):
    levels, counts = np.unique(col, return_counts=True)
    return levels, counts

def barplot(levels, counts, col):
    plt.bar(levels, counts, color="skyblue")
    plt.xlabel(f"{levels}")
    plt.ylabel(f"{counts}")
    plt.title(f"Frequency Count of {col} Levels")
    plt.show()

levels, counts = get_counts(X_train["bp_high"])
barplot(levels, counts, X_train["bp_high"])