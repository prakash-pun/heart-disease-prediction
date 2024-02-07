# all library import
import os
import pandas as pd

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "data/cardio_data_raw.csv")

df = pd.read_csv(file_path, sep=";")

print(df.head(5))
print(df.info())
print(df.isnull().head(5))
print(df["age"].isnull())
print(df["age"].isin([0,1]).all())
