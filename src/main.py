from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from models.train_model import TrainModel

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
model = TrainModel(X_train, X_test, y_train, y_test)

# SVM
svm_results = model.svm_model()
print("SVM Model:", svm_results)


# Logistic Regression
result_lr = model.logistic_regression_model()
print("Logistic Regression:", result_lr)


# Random Forest
results_rf = model.random_forest_model()
print("Random Forest Model: ", results_rf)


# XGBoost
metrics = model.xg_boost()
print("XGBoost_CLF ", metrics)


# Gradient Bosting Machine
gbm = model.gbm_model()
print("Gradient Boosting: ", gbm)
