# all library import
import pandas as pd
from utils import get_csv_file

path = get_csv_file("cleaned_data.csv")

df = pd.read_csv(path)

print(df.head(5))
print(df.info())
print(df.isnull().head(5))
print(df.isnull().sum())
print(df["age"].isnull())
print(df["age"].isin([0,1]).all())
print(df.describe())
