from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from rp_gbm_model import rp_gbm_model

from utils import generate_table
import pandas as pd
from model_tuners import train_lime_explainer, explain_prediction

X_train, X_test, y_train, y_test = split_data()

#Data Increment
X_train = pd.concat([X_train, X_train], ignore_index=False)
y_train = pd.concat([y_train, y_train], ignore_index=False)

# Data Filling
filled_x_train = fill_data(data_frame=X_train)
filled_x_test = fill_data(data_frame=X_test)

# Data Scaling
scaled_train_data = scale_minmax(filled_x_train)
scaled_test_data = scale_minmax(filled_x_test)

# Features
X_train = extract_feature(data_frame=scaled_train_data, y_train=y_train)
X_test = extract_feature(data_frame=scaled_test_data, y_train=y_test)

Model_Evaluation = (X_train, X_test, y_train, y_test)
print("Gradient Boosting:", Morp_gbm_modeldel_Evaluation)
# Train Models
model = TrainModel(X_train, X_test, y_train, y_test)

# Logistic Regression
result_lr = model.logistic_regression_model()
print("Logistic Regression:", result_lr)

# XGBoost
xg_boost = model.xg_boost()
print("XGBoost_CLF ", xg_boost)

#Performing lime on Xgboost
feature_names=X_train.columns.tolist()
sample_index = 0
# Train a LIME explainer
explainer = train_lime_explainer(X_train, feature_names)

# Explain a prediction
sample = X_test[sample_index]
explanation = explain_prediction(explainer, sample, model.xgboost, len(feature_names))

# Display the explanation
explanation.show_in_notebook()



#Gradient Bosting Machine
gbm = model.gbm_model()
print("Gradient Boosting: ", gbm)

#performing Lime on Gradient Boosting technique

feature_names=X_train.columns.tolist()
sample_index = 5

# Training a LIME explainer for gbm Model
rp_gbm_explainer = train_lime_explainer(X_train, feature_names)

# Explain a prediction
rp_gbm_sample = X_test[sample_index]
rp_gbm_explanation = explain_prediction(rp_gbm_explainer, rp_gbm_sample, model.gbm_model, len(feature_names))

explanation.show_in_notebook()


metrics = {
    "Logistic Regression Train": list(result_lr[0]),
    "Logistic Regression Test": list(result_lr[1]),
    "XGBoost_CLF Train": list(xg_boost[0]),
    "XGBoost_CLF Test": list(xg_boost[1]),
    "Gradient Boosting Train": list(gbm[0]),
    "Gradient Boosting Test": list(gbm[1]),
}

generate_table(metrics)