import pandas as pd
from data_loading import DataInitializer
from data_preprocessing import DataProcessor
from feature_engineering import FeatureEngines
from models.read_dump_model import DumpTrainModel
from model_tuners import ModelTuning
from models.feature_importance_analysis import FeatureImportanceAnalysis
from visualize import Visualizer

# Initializing Modules
reader = DataInitializer()
processor = DataProcessor()
extractor = FeatureEngines()
plotter = Visualizer()

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
X_train = extractor.extract_feature(
    data_frame=scaled_train_data, y_train=y_train)

train_columns = list(X_train.columns)
X_test = scaled_test_data[train_columns]

# MODELS
model = DumpTrainModel(X_train, X_test, y_train, y_test)

# Logistic Regression
result_lr = model.logistic_regression_model()

# XGBoost
xg_boost = model.xg_boost()

plotter.plot_roc(
    fpr=xg_boost["test"]["fpr"], tpr=xg_boost["test"]["tpr"], auc_score=xg_boost["test"]["roc_auc"])
plotter.plot_confusion_matrix(xg_boost["test"]["conf_matrix"])

# Gradient Bosting Machine
gbm = model.gbm_model()

reader.generate_table(result_lr, xg_boost, gbm)

# EXPLAINERS AND TUNERS
feature_names = X_train.columns.tolist()
tuners = ModelTuning(X_train, feature_names)

# Parameters for LIME
sample_index = 0
sample = X_test.values[sample_index]


# Explanation
explanation = tuners.explainer.explain_instance(
    sample, xg_boost["predict"].predict_proba, num_features=len(feature_names))
# print(explanation.as_list())

# Plotting
# tuners.plot_feature_importance(
#     result_lr, file_name='log_plot')
# tuners.plot_feature_importance(
#     xg_boost, file_name='xgb_plot')
# tuners.plot_feature_importance(gbm, file_name='gbm_plot')

# feature_importance_analysis followed by permutation analysis
model_files = {
    "Logistic Regression": "logistic_model.joblib",
    "XGBoost": "xg_boost_model.joblib",
    "Gradient Boosting": "gbm_model.joblib"
}
feature_importance_analysis = FeatureImportanceAnalysis(
    model_files, X_test, y_test)
feature_importance_analysis.plot_feature_importance()
feature_importance_analysis.permutation_importance_analysis()
