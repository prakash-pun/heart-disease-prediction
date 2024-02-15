import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import get_csv_file

csv_file = get_csv_file("merged_train_data.csv") 

df_framedata = pd.read_csv(csv_file)

print(df_framedata.describe())

X = df_framedata.drop(columns=["id", "cardio"])
id_cardio = df_framedata[['id', 'cardio']]
print(id_cardio.head(1))

#adding a new "id" column
df_framedata['id'] = range(1, len(df_framedata) + 1)

# with Standard Scaling
standard_scaler = StandardScaler()
scaled_data = standard_scaler.fit_transform(X)
scaled_data_frame = pd.DataFrame(scaled_data, columns=X.columns)

# Concatenate scaled_data_frame with id_cardio
result_df = pd.concat([id_cardio, scaled_data_frame], axis=1)

# Save the result to a new CSV file
result_df.to_csv("data/scaled_standard_dataset.csv", index=False)

print('Successfully saved scaled data with id and cardio columns.')

print(result_df.head(1))