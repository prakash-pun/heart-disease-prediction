import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Visualizer:
    
    def __init__(self):
        pass
    
    def get_counts(self, col):
        levels, counts = np.unique(col, return_counts=True)
        return levels, counts

    def match_index(self, data_frame, target):
        target_df = pd.DataFrame(target, columns=['cardio'])
        target_df.index = data_frame.index
        return target_df

    def barplot(self, data_frame):
        sns.set(style="whitegrid", palette="viridis")
        for col in data_frame.columns:
            plt.figure(figsize=(10, 6))
            levels, counts = self.get_counts(data_frame[col])
            plt.bar(levels, counts)
            plt.xlabel(f"{col}", fontsize=14, fontweight='bold')
            plt.ylabel("Counts", fontsize=14, fontweight='bold')
            plt.title(f"Distribution Count of {col} Levels", fontsize=16, fontweight='bold')
            plt.xticks(rotation=45)
            plt.show()

    def histplot(self, data_frame):
        sns.set(style="whitegrid")
        for col in data_frame.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data_frame[col], bins="auto", kde=True, color="lightblue")
            plt.title(f"Histogram of {col}", fontsize=16, fontweight='bold')
            plt.xlabel(f"{col}", fontsize=14, fontweight='bold')
            plt.ylabel("Frequency", fontsize=14, fontweight='bold')
            plt.show()

    def scatter(self, data_frame, target):
        sns.set(style="whitegrid")
        for col in data_frame.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data_frame, x=col, y=target, color="purple", alpha=0.7)
            plt.title(f"Scatter plot of {col} vs cardio", fontsize=16, fontweight='bold')
            plt.xlabel(f"{col}", fontsize=14, fontweight='bold')
            plt.ylabel("Cardio", fontsize=14, fontweight='bold')
            plt.show()

    def heatmap(self, data_frame, target, method="spearman"):
        sns.set(style="white")
        data = pd.concat([data_frame, target], axis=1)
        correlation = data.corr(method=method).round(2)
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='magma', cbar=False)
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.show()

    def plot_roc(self, fpr, tpr, auc_score):
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
        plt.xlabel('Predicted labels', fontsize=14, fontweight='bold')
        plt.ylabel('True labels', fontsize=14, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.show()
