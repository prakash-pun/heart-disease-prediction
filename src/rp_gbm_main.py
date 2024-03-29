from feature_extraction import extract_feature
from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from rp_gbm_model import rp_gbm_model
from utils import generate_table

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

Model_Evaluation = (X_train, X_test, y_train, y_test)
print("Gradient Boosting:", Morp_gbm_modeldel_Evaluation)

metrics = {

    "Gradient Boosting Train": list(Model_Evaluation[0]),
    "Gradient Boosting Test": list(Model_Evaluation[1]),
}

generate_table(metrics)