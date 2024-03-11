import os
import pandas as pd

def get_csv_file(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_dir, 'data')
    file_path = os.path.join(data_dir, f"{file_name}")

    return file_path


def get_data(file = "cleaned_data.csv"):
    """
    Read CSV and return the data frame
    Parameters
    ------------
    file: string
    """
    path = get_csv_file(file)
    df = pd.read_csv(path, index_col=0)

    return df
