import pandas as pd
from utils import get_csv_file

file_path = get_csv_file("cleaned_data.csv")

df = pd.read_csv(file_path)

print(df.info())
print(df.isnull().sum())
print(df["age"].isnull())
print(df["age"].isin([0,1]).all())
