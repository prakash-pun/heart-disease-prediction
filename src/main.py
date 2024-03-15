from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
<<<<<<< HEAD
from models.train_model import TrainModel
=======
from models.train_model import svm_model
from models.train_model_logreg import LogisticRegression_model1, LogisticRegression_model2
from models.random_forest import random_forest_model, ramdom_forest_recall
>>>>>>> ab77793c7fc3c0cf0f32bcbc7b0b3abc5efffbaa

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

#Train Models
model = TrainModel(X_train, X_test, y_train, y_test)

#SVM
svm_results= model.svm_model()
print("SVM Model:", svm_results)


<<<<<<< HEAD
# Precision
start2 = time.time()
precision_log = LogisticRegression_model2(X_train, X_test, y_train, y_test)
end2 = time.time()
print(precision_log)
print(end2-start2)

start=time.time()
recall=random_forest_recall(X_train, X_test, y_train, y_test)
end=time.time()
print(recall)
print(end-start)
<<<<<<< HEAD
=======
#Logistic Regression
result_lr = model.logistic_regression_model()
print("Logistic Regression:", result_lr)


#Random Forest
results_rf = model.random_forest_model()
print("Random Forest Model: ", results_rf)


#XGBoost
metrics = model.xg_boost()
print("XGBoost_CLF ", metrics)
>>>>>>> 124fb6f4b39a00f24531e30f4770559b65bac919
=======
>>>>>>> ab77793c7fc3c0cf0f32bcbc7b0b3abc5efffbaa
