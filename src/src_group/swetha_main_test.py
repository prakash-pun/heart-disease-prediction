from src.data_loading import DataInitializer
from src.data_preprocessing import DataProcessor
from src.feature_engineering import FeatureEngines
from src.visualize import Visualizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.read_dump_model import DumpTrainModel
from models.feature_importance_analysis import FeatureImportanceAnalysis
from utils.runner import run_streamlit
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel



# Initializing Modules
reader = DataInitializer()
processor = DataProcessor()
extractor = FeatureEngines()
plotter = Visualizer()

# Data Preparation
X_train, X_test, y_train, y_test = reader.split_data()

# Data Filling
filled_x_train = processor.fill_data(data_frame=X_train)
filled_x_test = processor.fill_data(data_frame=X_test)

# Univariate Analysis
plotter.barplot(X_train)

# Bivariate Analysis
plotter.scatter(X_train, y_train)

# MultiCollinearity
plotter.heatmap(X_train, y_train)

# Data Scaling
scaled_train_data = extractor.scale_minmax(filled_x_train)
scaled_test_data = extractor.scale_minmax(filled_x_test)
# plotter.heatmap(scaled_train_data, y_train)

# Feature Extraction
X_train = extractor.extract_feature(
    data_frame=scaled_train_data, y_train=y_train)
plotter.heatmap(X_train, y_train)
train_columns = list(X_train.columns)
X_test = scaled_test_data[train_columns]

# MODELS
model = DumpTrainModel(X_train, X_test, y_train, y_test)

# Logistic Regression
result_lr = model.logistic_regression_model()

# XGBoost
xg_boost = model.xg_boost()

# Results
plotter.plot_roc(
    fpr=xg_boost["test"]["fpr"], tpr=xg_boost["test"]["tpr"], auc_score=xg_boost["test"]["roc_auc"])
plotter.plot_confusion_matrix(xg_boost["test"]["conf_matrix"])

# Gradient Bosting Machine
gbm = model.gbm_model()

reader.generate_table(result_lr, xg_boost, gbm)

# FEATURE IMPORTANCE
model_files = {
    "Logistic Regression": "logistic_model.joblib",
    "XGBoost": "xg_boost_model.joblib",
    "Gradient Boosting": "gbm_model.joblib"
}
feature_importance_analysis = FeatureImportanceAnalysis(
    model_files, X_test, y_test)
feature_importance_analysis.plot_feature_importance()
feature_importance_analysis.permutation_importance_analysis()




# most important features
feature_importances = feature_importance_analysis.xg_boost()["feature_importance"]
top_features = [feature_name for _, feature_name in sorted(zip(feature_importances, feature_importance_analysis.xg_boost()["feature_names"]), reverse=True)][:3]

X_train_new = X_train.copy()
X_train_new['top_features_combined'] = X_train[top_features[0]] * X_train[top_features[1]] * X_train[top_features[2]]

X_test_new = X_test.copy()
X_test_new['top_features_combined'] = X_test[top_features[0]] * X_test[top_features[1]] * X_test[top_features[2]]

# Tune the XGBoost
xgb = XGBClassifier()
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_new, y_train)
best_xgb_model = grid_search.best_estimator_




# less important features
feature_importances = feature_importance_analysis.xg_boost()["feature_importance"]
less_important_features = [feature_name for _, feature_name in sorted(zip(feature_importances, feature_importance_analysis.xg_boost()["feature_names"]))][:3]

xgb = XGBClassifier()
selector = SelectFromModel(xgb, prefit=True, threshold='mean')
X_train_reduced = selector.transform(X_train)
X_test_reduced = selector.transform(X_test)

# Tune the XGBoost model
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_reduced, y_train)
best_xgb_model = grid_search.best_estimator_


print("Running Streamlit...")
run_streamlit()