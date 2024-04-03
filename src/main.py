import pandas as pd
from utils import DataInitializer
from data_preprocessing import DataProcessor
from feature_engineering import FeatureEngines
from models.train_model import TrainModel
from model_tuners import ModelTuning

#Initializing Modules
reader = DataInitializer()
processor = DataProcessor()
extractor = FeatureEngines()

# Data Preparation
X_train, X_test, y_train, y_test = reader.split_data()

# Data Increment
X_train = pd.concat([X_train, X_train], ignore_index=False)
y_train = pd.concat([y_train, y_train], ignore_index=False)

# Data Filling
filled_x_train = processor.fill_data(data_frame=X_train)
filled_x_test = processor.fill_data(data_frame=X_test)

# Data Scaling
scaled_train_data = extractor.scale_minmax(filled_x_train)
scaled_test_data = extractor.scale_minmax(filled_x_test)

# Feature Extraction
X_train = extractor.extract_feature(data_frame=scaled_train_data, y_train=y_train)

train_columns = list(X_train.columns)
X_test = scaled_test_data[train_columns]

#%% MODELS
model = TrainModel(X_train, X_test, y_train, y_test)

# Logistic Regression
result_lr = model.logistic_regression_model()
print("Logistic Regression:", result_lr)

# XGBoost
xg_boost = model.xg_boost()
print("XGBoost_CLF ", xg_boost)

# Gradient Bosting Machine
gbm = model.gbm_model()
print("Gradient Boosting: ", gbm)

# Results
metrics = {
    "Logistic Regression Train": list(result_lr["train"]),
    "Logistic Regression Test": list(result_lr["test"]),
    "XGBoost_CLF Train": list(xg_boost["train"]),
    "XGBoost_CLF Test": list(xg_boost["test"]),
    "Gradient Boosting Train": list(gbm["train"]),
    "Gradient Boosting Test": list(gbm["test"]),
}

reader.generate_table(metrics)

#%% EXPLAINERS AND TUNERS

feature_names = X_train.columns.tolist()
tuners = ModelTuning(X_train, feature_names)

# Parameter for LIME
sample_index = 0
sample = X_test.values[sample_index]
pred_prob = xg_boost["predict"].predict_proba(sample.reshape(1, -1))

predict_fn = lambda x: pred_prob

# Explaination
explanation = tuners.explain_prediction(sample, predict_fn, num_features=len(feature_names))
print(explanation.as_list())

# Plotting
tuners.plot_feature_importance(result_lr, result_lr["feature_names"], file_name='log_plot')
tuners.plot_feature_importance(xg_boost, xg_boost["feature_names"], file_name='xgb_plot')
tuners.plot_feature_importance(gbm, gbm["feature_names"], file_name='gbm_plot')
