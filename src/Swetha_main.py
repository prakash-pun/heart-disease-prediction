import warnings
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame")

from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from models.Swetha_KNNmodel import Swetha_KNNmodel
from models.Swetha_KNNmodel import Swetha_DecisionTreeModel

def main():
    # Split the dataset into train and test datasets
    X_train, X_test, y_train, y_test = split_data()

    # Data Filling
    filled_x_train = fill_data(data_frame=X_train)
    filled_x_test = fill_data(data_frame=X_test)

    # Data Scaling
    scaled_train_data = scale_minmax(filled_x_train)
    scaled_test_data = scale_minmax(filled_x_test)

    # Features
    X_train = extract_feature(data_frame=scaled_train_data, y_train=y_train)
    X_test = extract_feature(data_frame=scaled_test_data, y_train=y_test)

    # Train Models
    model = Swetha_KNNmodel(X_train, X_test, y_train, y_test)

    # KNN model
    knn_results = model
    print("KNN Model: ", knn_results)

    model = Swetha_DecisionTreeModel(X_train, X_test, y_train, y_test)

    # Decision Tree Model
    Dec_Tree_results = model
    print("Decision Tree Model: ", Dec_Tree_results)

    return model

if __name__ == "__main__":
    trained_model = main()
