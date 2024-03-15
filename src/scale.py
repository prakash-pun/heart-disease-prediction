import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_minmax(data_frame):    
    
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(data_frame)
    scaled_data_frame = pd.DataFrame(scaled_data, columns=data_frame.columns)
    scaled_data_frame.index = data_frame.index

    return scaled_data_frame


def scale_standard(data_frame):
    
    standard_scaler = StandardScaler()
    scaled_data = standard_scaler.fit_transform(data_frame)
    scaled_data_frame = pd.DataFrame(scaled_data, columns=data_frame.columns)
    scaled_data_frame.index = data_frame.index

    return scaled_data_frame
