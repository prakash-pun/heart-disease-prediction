from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from models.gaurav_model import Lsvc,train_naive_bayes,calc_metric

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


#Linear svc
y_pred_lsvc, y_prob_lsvc = Lsvc(X_train, X_test, y_train, y_test)

#calculate metrics
metrics1 = calc_metric(y_test, y_pred_lsvc, y_prob_lsvc)
print(f"metrics for Lsvc: {metrics1}" )

# Train the Naive Bayes model
y_pred_NB,y_prob_NB = train_naive_bayes(X_train, X_test, y_train)

#calculate metrics
metrics2 = calc_metric(y_test, y_pred_NB, y_prob_NB)
print(f"metrics for NB: {metrics2}" )
