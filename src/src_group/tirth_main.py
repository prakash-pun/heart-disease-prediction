from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from models.train_model import TrainModel
from utils import generate_table
from model_tuners import train_lime_explainer, explain_prediction
from sklearn.linear_model import LogisticRegression

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

# Logistic Regression
logistic_regression_model = model.logistic_regression_model()
print("Logistic Regression:", logistic_regression_model)

# Performing lime on Logistic regression
feature_names = X_train.columns.tolist()
sample_index = 0

# Verify X_train and feature_names are defined and have the correct format
print("X_train shape:", X_train.shape)  # Check shape or type
print("Feature names:", feature_names)   # Check if feature names are correctly defined

# Training lime explainer
explainer = train_lime_explainer(X_train, feature_names)

# Select an instance for explanation
sample = X_test.iloc[sample_index]
# Convert the pandas Series to a numpy array
sample_array = sample.values.reshape(1, -1)

# Define and train the logistic regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Verify the type of the model object
print(type(logistic_regression_model))

# Inspect the attributes and methods of the model object
print(dir(logistic_regression_model))

# Explain the instance using LIME
explanation = explain_prediction(explainer, sample_array, logistic_regression_model, len(feature_names))


# Display the explanation
explanation.show_in_notebook()

# XGBoost
xg_boost = model.xg_boost()
print("XGBoost_CLF ", xg_boost)

# Gradient Boosting Machine
gbm = model.gbm_model()
print("Gradient Boosting: ", gbm)

metrics = {
    "Logistic Regression Train": list(logistic_regression_model[0]),
    "Logistic Regression Test": list(logistic_regression_model[1]),
    "XGBoost_CLF Train": list(xg_boost[0]),
    "XGBoost_CLF Test": list(xg_boost[1]),
    "Gradient Boosting Train": list(gbm[0]),
    "Gradient Boosting Test": list(gbm[1]),
}

generate_table(metrics)
