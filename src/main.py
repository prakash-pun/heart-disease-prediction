from data_loading import DataInitializer
from data_preprocessing import DataProcessor
from feature_engineering import FeatureEngines
from models.read_dump_model import DumpTrainModel
from models.feature_importance_analysis import FeatureImportanceAnalysis
from visualize import Visualizer

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
plotter.heatmap(scaled_train_data, y_train)

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


# DASHBOARD
