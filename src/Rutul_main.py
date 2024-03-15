
from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from Rutul_svm_model import rp_svm_model

X_train, X_test, y_train, y_test = split_data()

filled_x_train = fill_data(data_frame=X_train)
filled_x_test = fill_data(data_frame=X_test)


scaled_train_data = scale_minmax(filled_x_train)
scaled_test_data = scale_minmax(filled_x_test)

X_train = extract_feature(data_frame=scaled_train_data, y_train=y_train)
X_test = extract_feature(data_frame=scaled_test_data, y_train=y_test)


#Model Evaluation

Model_Evaluation = rp_svm_model(X_train, X_test, y_train, y_test)
#1. Accuracy
Accuracy_Evaluation = Model_Evaluation[:2]
print("Accuracy of training and test data is:",Accuracy_Evaluation)
#2. Precision
Precision_Evaluation = Model_Evaluation[2:4]
print("Precision of training and test data is:",Precision_Evaluation)
#3. F1_Score
f1_score_Evaluation = Model_Evaluation[4:6]
print("F1 Score of training and test data is:",f1_score_Evaluation)
#4. Recall
recall_Evaluation = Model_Evaluation[6:]
print("Recall Score of training and test data is:",recall_Evaluation)