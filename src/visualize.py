import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    
    def get_counts(col):
        levels, counts = np.unique(col, return_counts=True)
        return levels, counts

    def barplot(levels, counts, col):
        plt.bar(levels, counts, color="skyblue")
        plt.xlabel(f"{levels}")
        plt.ylabel(f"{counts}")
        plt.title(f"Frequency Count of {col} Levels")
        plt.show()

    def histplot(col):
        plt.figure(figsize=(8, 8))
        sns.displot(col, bins="auto")
        plt.show()

    def scatter(data_frame, col):
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data_frame, x=data_frame.col, y=data_frame.cardio)
        plt.show()

    def heat_map(data, data2, method="spearman"):
        
        correlation = data.corrwith(data2, method=method).round(2)
        plt.figure(figsize=(15, 15))
        sns.heatmap(correlation, annot=True)
        plt.show()

    def plot_roc(fpr, tpr):
        # Plot ROC curve
        plt.plot(fpr, tpr, color='blue', label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='red',
             linestyle='--', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

