import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


data_frame = pd.read_csv("../data/merged_train_data.csv")

#assigning the columns to be excluded in a variable

Z = data_frame.loc[:, data_frame.columns.isin(
    ["id","gender", "cardio", "cholesterol", "gluc", "diabetic", "smoke", "alco", "active", "height_m", "cholesterol_1",
     "cholesterol_2", "cholesterol_3", "gluc_1", "gluc_2", "gluc_3", "diabetic_1", "diabetic_2", "diabetic_3"])]

#excluding the columns using a variable

X = data_frame.loc[:, ~data_frame.columns.isin(
    ["id","gender", "cardio", "cholesterol", "gluc", "diabetic", "smoke", "alco", "active", "height_m", "cholesterol_1",
     "cholesterol_2", "cholesterol_3", "gluc_1", "gluc_2", "gluc_3", "diabetic_1", "diabetic_2", "diabetic_3"])]
y = data_frame["cardio"]

# with MinMaxScaler
min_max_scaler = MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(X)
scaled_data_frame = pd.DataFrame(scaled_data, columns=X.columns)

print(scaled_data_frame.describe())

scaled_data_frame['cardio'] = y

#scaled_data_frame.to_csv("scaled_without_bin.csv")

#adding the excluded columns in csv

result_df = pd.concat([Z, scaled_data_frame], axis=1)

result_df.to_csv("../data/scaled_without_bin.csv", index=False)

print(scaled_data_frame.head(1))
