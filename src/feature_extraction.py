def extract_feature(data_frame, y_train):
    df = data_frame.copy(deep=True)
    raw_correl = df.corrwith(y_train, method='spearman').round(2)
    
    threshold = 0.06
    raw_cols = raw_correl[abs(raw_correl) > threshold].index.tolist()
    raw_features = df[raw_cols]

    return raw_features
