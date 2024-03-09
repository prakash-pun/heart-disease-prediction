import time
from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from models.random_forest import random_forest_model

X_train, X_test, y_train, y_test = split_data()

# fill and merge train dataset
filled_x_train = fill_data(data_frame=X_train, file_name="merged_train_data.csv")
filled_x_test = fill_data(data_frame=X_test, file_name="merged_test_data.csv")
# scale
scaled_train_data = scale_minmax(filled_x_train)
scaled_test_data = scale_minmax(filled_x_test)

X_train = extract_feature(data_frame=scaled_train_data, y_train=y_train)
X_test = extract_feature(data_frame=scaled_test_data, y_train=y_test)

start = time.time()

precision = random_forest_model(X_train, X_test, y_train, y_test)

end = time.time()


print(precision)
print(end-start)