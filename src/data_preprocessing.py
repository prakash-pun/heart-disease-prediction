import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class DataProcessor:
    
    def __init__(self, threshold=0.18):
        self.threshold = threshold
        self.encoder = OneHotEncoder()

    def fill_data(self, data_frame):
        df = data_frame.copy(deep=True)
        m_bp_lo = df.loc[:, 'bp_lo'].median()
        m_round = round(m_bp_lo, -1)
        df.fillna({"bp_lo": m_round}, inplace=True)
        df['height_m'] = df['height'] / 100
        df['bmi'] = df['weight'] / (df['height_m'] ** 2)
        df['bmi'] = df['bmi'].round()

        ohreq = df[['cholesterol', 'gluc', 'diabetic']]
        oh_encoded = self.encoder.fit_transform(ohreq)
        oh_df = pd.DataFrame(oh_encoded.toarray(), columns=self.encoder.get_feature_names_out(
            ['cholesterol', 'gluc', 'diabetic']))
        oh_df.index = df.index
        df = pd.concat([df, oh_df], axis=1)

        return df

# Example usage:
# processor = DataProcessor()
# processed_data = processor.fill_data(data_frame)
# selected_features = processor.extract_feature(processed_data, y_train)
