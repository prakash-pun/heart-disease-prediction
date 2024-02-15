import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


data_frame = pd.read_csv("merged_train_data.csv")

X = data_frame.loc[:, ~data_frame.columns.isin(
    ["gender", "cardio", "cholesterol", "gluc", "diabetic", "smoke", "alco", "active", "height_m", "cholesterol_1",
     "cholesterol_2", "cholesterol_3", "gluc_1", "gluc_2", "gluc_3", "diabetic_1", "diabetic_2", "diabetic_3"])]
y = data_frame["cardio"]

# with MinMaxScaler
min_max_scaler = MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(X)
scaled_data_frame = pd.DataFrame(scaled_data, columns=X.columns)

print(scaled_data_frame.describe())

scaled_data_frame['cardio'] = y

scaled_data_frame.to_csv("scaled_without_bin.csv")

print(scaled_data_frame.head(1))
