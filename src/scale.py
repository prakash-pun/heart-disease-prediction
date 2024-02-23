import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_csv_file

def scale_minmax():
    csv_file = get_csv_file("merged_train_data.csv") 
    data_frame = pd.read_csv(csv_file)
    
    X = data_frame.drop(columns=["id", "cardio"])

    y = data_frame["cardio"]
    id = data_frame["id"]

    # with MinMaxScaler
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(X)
    scaled_data_frame = pd.DataFrame(scaled_data, columns=X.columns)

    scaled_data_frame['cardio'] = y
    scaled_data_frame.insert(loc=0, column='id', value=id)

    scaled_data_frame.to_csv("data/scaled_dataset.csv", index=False)

    return scaled_data_frame


def scale_standard():
    csv_file = get_csv_file("merged_train_data.csv") 
    df_framedata = pd.read_csv(csv_file)

    X = df_framedata.drop(columns=["id", "cardio"])
    id_cardio = df_framedata[['id', 'cardio']]

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

    return result_df


def scale_without_bin():
    data_frame = pd.read_csv("../data/merged_train_data.csv")

    Z = data_frame.loc[:, data_frame.columns.isin(
        ["id","gender", "cardio", "cholesterol", "gluc", "diabetic", "smoke", "alco", "active", "height_m", "cholesterol_1",
        "cholesterol_2", "cholesterol_3", "gluc_1", "gluc_2", "gluc_3", "diabetic_1", "diabetic_2", "diabetic_3"])]

    X = data_frame.loc[:, ~data_frame.columns.isin(
        ["id","gender", "cardio", "cholesterol", "gluc", "diabetic", "smoke", "alco", "active", "height_m", "cholesterol_1",
        "cholesterol_2", "cholesterol_3", "gluc_1", "gluc_2", "gluc_3", "diabetic_1", "diabetic_2", "diabetic_3"])]
    y = data_frame["cardio"]

    # with MinMaxScaler
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(X)
    scaled_data_frame = pd.DataFrame(scaled_data, columns=X.columns)

    scaled_data_frame['cardio'] = y

    result_df = pd.concat([Z, scaled_data_frame], axis=1)

    result_df.to_csv("../data/scaled_without_bin.csv", index=False)

    return result_df
