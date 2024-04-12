import pandas as pd
import lime
import lime.lime_tabular
from feature_extraction import extract_feature
from models.train_model import TrainModel
from split_dataset import split_data
from utils import generate_table
from fill_data import fill_data
from scale import scale_minmax

X_train, X_test, y_train, y_test = split_data()

# Data Increment
# X_train = pd.concat([X_train, X_train], ignore_index=False)
# y_train = pd.concat([y_train, y_train], ignore_index=False)

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
result_lr = model.logistic_regression_model()
print("Logistic Regression:", result_lr)


# Performing lime on Xgboost
feature_names = X_train.columns.tolist()

# Train a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=feature_names,
                                                   class_names=['0', '1'])


# Explain a prediction
sample = X_test.values[1]
for i in range(5):
    explanation = explainer.explain_instance(
        sample, result_lr["predict"].predict_proba, num_features=len(feature_names))
# Display the explanation
    print(i, explanation.as_list())


metrics = {
    "Logistic Regression Train": list(result_lr["train"]),
    "Logistic Regression Test": list(result_lr["test"]),
}

generate_table(metrics)
