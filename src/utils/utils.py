import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix


def metrics(y_test, predictions, proba):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, proba)

    fpr, tpr, thresholds = roc_curve(y_test, proba)

    conf_matrix = confusion_matrix(y_test, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "conf_matrix": conf_matrix
    }


class DataInitializer:

    def __init__(self):
        self.data_dir = self.get_data_directory()

    def get_data_directory(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

        return os.path.join(project_dir, 'data')

    def get_csv_file(self, file_name):

        return os.path.join(self.data_dir, file_name)

    def get_data(self, file="cleaned_data.csv"):
        path = self.get_csv_file(file)

        return pd.read_csv(path, index_col=0)

    def split_data(self, file="cleaned_data.csv"):
        data_frame = self.get_data(file)
        X = data_frame.drop("cardio", axis=1)
        y = data_frame["cardio"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def generate_table(self, metrics):
        df = pd.DataFrame(data=metrics, index=[
                          "Accuracy", "Precision", "Recall", "F1 Score", "ROC Auc"]).T
        pd.set_option('display.float_format', '{:.15f}'.format)
        print(df)
