from sklearn.model_selection import train_test_split 
from utils import get_data


def split_data():
    df = get_data()

    X = df.drop("cardio", axis = 1)
    y = df["cardio"] # target

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    # # saving the train data
    # X_train.to_csv("data/train_data.csv", index=False)
    # X_test.to_csv("data/test_data.csv", index=False)
    # y_train.to_csv("data/train_data_labels.csv", index=False, header=True)
    # y_test.to_csv("data/test_data_labels.csv", index=False, header=True)

    # return X_train, X_test, y_train, y_test
    return train, test

