form feature_extraction import extract_feature
from split_dataset import split_dataset
from fill_data import fill_data
from scale import scale_minmax
from models.Janki_KNNmodel import Janki_KNNmodel

X_train, X_test, y_train, y_test = split_data()

#fill and merge train dataset
filled_x_train = fill_data(data_frame=X_train)
filled_x_test = fill_data(data_frame=X_test)

#scale
scaled_train_data = scale_minmax(filled_x_train)
scaled_test_data = scale_minmax(fillled_x_test)

X_train = extract_feature(data_frame=scaled_train_data, y_train=y_train)
X_test = extract_feature(data_frame=scaled_test_data, y_train=y_test)

result = KNN_model(X_train, X_test, y_train, y_test)
print(result)


