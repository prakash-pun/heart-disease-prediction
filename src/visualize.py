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
        for col in data_frame.columns:
            levels, counts = self.get_counts(data_frame[col])
            plt.bar(levels, counts, color="skyblue")
            plt.xlabel(f"{col}")
            plt.ylabel("Counts")
            plt.title(f"Distribution Count of {col} Levels")
            plt.show()

    def histplot(self, data_frame):
        for col in data_frame.columns:
            plt.figure(figsize=(8, 8))
            sns.histplot(data_frame[col], bins="auto")
            plt.title(f"Histogram of {col}")
            plt.show()

    def scatter(self, data_frame,target):
        for col in data_frame.columns:
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=data_frame, x=col, y=target["cardio"])
            plt.title(f"Scatter plot of {col} vs cardio")
            plt.show()

    def heatmap(self, data_frame, target, method = "spearman"):
        data = pd.concat([data_frame, target], axis=1)
        correlation = data.corr(method = method).round(2)
        plt.figure(figsize=(15, 15))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=False)
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_roc(self, fpr, tpr, auc_score):
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        
    def plot_confusion_matrix(conf_matrix):

        # y_pred = model.predict(X_test)
        # conf_matrix = confusion_matrix(y_test, y_pred)
    
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

