import time
from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from models.train_model import TrainModel

X_train, X_test, y_train, y_test = split_data()

#fill and merge train dataset
filled_x_train = fill_data(data_frame=X_train)
filled_x_test = fill_data(data_frame=X_test)

# scale
scaled_train_data = scale_minmax(filled_x_train)
scaled_test_data = scale_minmax(filled_x_test)

X_train = extract_feature(data_frame=scaled_train_data, y_train=y_train)
X_test = extract_feature(data_frame=scaled_test_data, y_train=y_test)

#svm
start = time.time()
model = TrainModel(X_train, X_test, y_train, y_test)
accuracy, f1, precision = model.svm_model()
end = time.time()
print("SVM Model:", accuracy, f1, precision)
print(end-start)

start2 = time.time()
result_lr = model.logistic_regression_model()
end2 = time.time()
print("Logistic Regression:", result_lr)
print(end2-start2)

start = time.time()
result = model.random_forest_model()
end = time.time()
print("Random Forest Model: ", result)
print(end-start)
