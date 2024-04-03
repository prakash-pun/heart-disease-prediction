import pandas as pd
from model_tuners import ModelTuning

from feature_extraction import extract_feature
from models.train_model import TrainModel
from split_dataset import split_data
from utils import generate_table
from fill_data import fill_data
from scale import scale_minmax
from feature_imp_analysis import plot_feature_importance, calculate_feature_importance


X_train, X_test, y_train, y_test = split_data()

# Data Increment
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
train_columns = list(X_train.columns)

X_test = scaled_test_data[train_columns]

# Train Models
model = TrainModel(X_train, X_test, y_train, y_test)

# Logistic Regression
result_lr = model.logistic_regression_model()
print("Logistic Regression:", result_lr)

# XGBoost
xg_boost = model.xg_boost()
print("XGBoost_CLF ", xg_boost)

# Performing lime on Xgboost
feature_names = X_train.columns.tolist()

# sample_index = 0
# # Train a LIME explainer
# explainer = train_lime_explainer(X_train, feature_names)
#
# # predict_fn = lambda x: model.predict(xgb.DMatrix(x))
#
# # Explain a prediction
# sample = X_test.values[sample_index]
#
#
# explanation = explainer.explain_instance(
#     sample, xg_boost["predict"].predict_proba, num_features=len(feature_names))
# # Display the explanation
# print(explanation.as_list())


# Create an instance of ModelTuning
model_tuner = ModelTuning(X_train, feature_names)

# Assuming xg_boost is an XGBoost model
sample_index = 0
sample = X_test.values[sample_index]
pred_prob = xg_boost["predict"].predict_proba(sample.reshape(1, -1))

predict_fn = lambda x: pred_prob

# Explain a prediction using the ModelTuning instance
explanation = model_tuner.explain_prediction(sample, predict_fn, num_features=len(feature_names))

# Display the explanation
print(explanation.as_list())

# Gradient Bosting Machine
gbm = model.gbm_model()
print("Gradient Boosting: ", gbm)

metrics = {
    "Logistic Regression Train": list(result_lr["train"]),
    "Logistic Regression Test": list(result_lr["test"]),
    "XGBoost_CLF Train": list(xg_boost["train"]),
    "XGBoost_CLF Test": list(xg_boost["test"]),
    "Gradient Boosting Train": list(gbm["train"]),
    "Gradient Boosting Test": list(gbm["test"]),
}

generate_table(metrics)

# Plotting
plot_feature_importance(result_lr, result_lr["feature_names"], file_name='log_plot')
plot_feature_importance(xg_boost, xg_boost["feature_names"], file_name='xgb_plot')
plot_feature_importance(gbm, gbm["feature_names"], file_name='gbm_plot')
