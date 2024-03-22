from sklearn.model_selection import train_test_split 
from utils import get_data
import pandas as pd


def split_data():
    """
    Split Cleaned Data to test and train data set
    Returns
    -------
    train
        dataset for train data
    test 
        dataset for test data
    """
    org_data_frame = get_data("cleaned_data.csv")
    
    data_frame = pd.concat([org_data_frame,org_data_frame], ignore_index=False)
    X = data_frame.drop("cardio", axis=1)
    y = data_frame["cardio"] 
    
    # target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test