import os
import pandas as pd


def get_csv_file(file_name):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, f"data/{file_name}")

    return file_path


def get_data():
    path = get_csv_file("cleaned_data.csv")
    df = pd.read_csv(path)

    return df


print("asdfasdf", get_csv_file("merged_train_data.csv"))