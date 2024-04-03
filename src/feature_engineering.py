import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureEngines:
    
    def __init__(self, threshold=0.18):
        self.threshold = threshold

    def scale_minmax(self, data_frame):    
        min_max_scaler = MinMaxScaler()
        scaled_data = min_max_scaler.fit_transform(data_frame)
        scaled_data_frame = pd.DataFrame(scaled_data, columns=data_frame.columns)
        scaled_data_frame.index = data_frame.index
        
        return scaled_data_frame

    def scale_standard(self, data_frame):
        standard_scaler = StandardScaler()
        scaled_data = standard_scaler.fit_transform(data_frame)
        scaled_data_frame = pd.DataFrame(scaled_data, columns=data_frame.columns)
        scaled_data_frame.index = data_frame.index
        
        return scaled_data_frame

    def extract_feature(self, data_frame, y_train):
        df = data_frame.copy(deep=True)
        raw_correl = df.corrwith(y_train, method='spearman').round(2)
        raw_cols = raw_correl[abs(raw_correl) > self.threshold].index.tolist()
        raw_features = df[raw_cols]
        
        return raw_features