from sklearn.model_selection import train_test_split 
from utils import get_data


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
    data_frame = get_data("cleaned_data.csv")

    X = data_frame.drop("cardio", axis=1)
    y = data_frame["cardio"]  # target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # # saving the train and test data
    # train.to_csv("data/train_data.csv", index=False, header=True)
    # test.to_csv("data/test_data.csv", index=False, header=True)

    return X_train, X_test, y_train, y_test
