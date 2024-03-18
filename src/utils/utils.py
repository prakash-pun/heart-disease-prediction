import os
import pandas as pd
import numpy as np


def get_csv_file(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_dir, 'data')
    file_path = os.path.join(data_dir, f"{file_name}")

    return file_path


def get_data(file="cleaned_data.csv"):
    """
    Read CSV and return the data frame
    Parameters
    ------------
    file: string
    """
    path = get_csv_file(file)
    df = pd.read_csv(path, index_col=0)

    return df


def generate_table(metrics):
    df = pd.DataFrame(
        data=metrics, index=["Accuracy", "Precision", "Recall", "F1 Score", "ROC Auc"])
    df = df.T
    pd.set_option('display.float_format', '{:.15f}'.format)
    print(df)
