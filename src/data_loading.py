import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataInitializer:

    def __init__(self):
        self.data_dir = self.get_data_directory()

    def get_data_directory(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(current_dir, '..'))
        path = os.path.join(project_dir, 'data')

        return path

    def get_csv_file(self, file_name):
        path = os.path.join(self.data_dir, file_name)

        return path

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

    def generate_table(self, logistic, xg_boost, gbm):
        metrics = {
            "Logistic Regression Train": logistic["train"],
            "Logistic Regression Test": logistic["test"],
            "XGBoost_CLF Train": xg_boost["train"],
            "XGBoost_CLF Test": xg_boost["test"],
            "Gradient Boosting Train": gbm["train"],
            "Gradient Boosting Test": gbm["test"],
        }

        keys_to_remove = ['fpr', 'tpr', 'conf_matrix']

        # Remove keys
        for model_data in metrics.values():
            for key in keys_to_remove:
                model_data.pop(key, None)

        # Create a new dictionary with model names as keys and list of metrix
        result_dict = {model: list(data.values())
                       for model, data in metrics.items()}

        df = pd.DataFrame(data=result_dict, index=[
                          "Accuracy", "Precision", "Recall", "F1 Score", "ROC Auc"]).T
        pd.set_option('display.float_format', '{:.15f}'.format)
        print(df)
