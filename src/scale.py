import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_csv_file

def scale_minmax():
    csv_file = get_csv_file("merged_train_data.csv") 

    data_frame = pd.read_csv(csv_file)

    print(data_frame.describe())

    X = data_frame.drop(columns=["id", "cardio"])
    y = data_frame["cardio"]
    id = data_frame["id"]

    # with MinMaxScaler
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(X)
    scaled_data_frame = pd.DataFrame(scaled_data, columns=X.columns)

    print(scaled_data_frame.describe())

    scaled_data_frame['cardio'] = y
    scaled_data_frame["id"] = id

    scaled_data_frame.to_csv("data/scaled_dataset.csv", index=[False])

    return scaled_data_frame


# TODO: Try Standard Scaler
# resuource
# https://medium.com/codex/why-scaling-your-data-is-important-1aff95ca97a2
