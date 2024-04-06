from utils import DataInitializer
from data_preprocessing import DataProcessor
from feature_engineering import FeatureEngines
from models.read_dump_model import DumpTrainModel
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import streamlit as st
from joblib import load, dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from models.dash import samefeature, calculate_and_add_bmi, preprocess_input_data, encode_input_data

di = DataInitializer()
dp = DataProcessor()
fe = FeatureEngines()

data_frame = di.get_data()

X_train, X_test, y_train, y_test = di.split_data()

# Data Increment
X_train = pd.concat([X_train, X_train], ignore_index=False)
y_train = pd.concat([y_train, y_train], ignore_index=False)


# Data Filling
filled_x_train = dp.fill_data(data_frame=X_train)
filled_x_test = dp.fill_data(data_frame=X_test)


# Data Scaling
min_max_scaler = MinMaxScaler()

# Fit and transform the training data
scaled_train_data = fe.scale_minmax(filled_x_train)
scaled_test_data = fe.scale_minmax(filled_x_test)



# Features
X_train = fe.extract_feature(data_frame=scaled_train_data, y_train=y_train)
train_columns = list(X_train.columns)
X_test = scaled_test_data[train_columns]

# metrics
def metrics(y_test, predictions, proba):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, proba)

    return accuracy, precision, recall, f_score, roc_auc

#xgboost
# params = {
#             'booster': ['gbtree', 'gblinear'],  # gblinear
#             'learning_rate': np.arange(0.01, 0.9, 0.01),
#             'n_estimators': range(50, 1000, 50),
#             'objective': ['binary:logistic'],
#             # ,'multi:softmax','multi:softprob','reg:logitstic'],  # Binary classification
#             'eval_metric': ['merror', 'logloss', 'auc']  # Evaluation metric
#             # 'subsample': np.arange(0.1, 0.9, 0.1),
#             # 'max_depth': range(2, 7)
#         }
# model = xgb.XGBClassifier()
# random_search = RandomizedSearchCV(estimator=model, param_distributions=params,n_iter=100, scoring="recall", cv=5, random_state=42, n_jobs=-1)
#
# random_search.fit(X_train, y_train)
# best_param = random_search.best_params_
#
# # XGB CLF
# xgb_clf = xgb.XGBClassifier(**best_param)
# xgb_clf.fit(X_train, y_train)
#
# # Make predictions
# train_predict = xgb_clf.predict(X_train)
# train_predictions_clf = (train_predict > 0.5).astype(int)
# train_proba = xgb_clf.predict_proba(X_train)
#
# predictions_clf = xgb_clf.predict(X_test)
# binary_predictions_clf = (predictions_clf > 0.5).astype(int)
# predict_proba = xgb_clf.predict_proba(X_test)
#
# # Calculate metrics
# result_train = metrics(y_train, train_predictions_clf, train_proba[:, 1])
# result = metrics(y_test, binary_predictions_clf,predict_proba[:, 1])
#
# # saving a model
# dump(xgb_clf,"../src/models/xg.model")

# model load
rf_classif = load("../src/models/xg.model")

# for checking if this is working or not
prediction = rf_classif.predict(X_test)
print(prediction)

st.title("heart disease :red[Prediction] :bar_chart: :chart_with_upwards_trend:")
st.markdown("Predict heart Type using different parameters")

tab1, tab2, tab3 = st.tabs(["Data :clipboard:", "Performance :weight_lifter:", "Local Performance :bicyclist:"])

with tab1:
    st.header("DATASET")
    st.write(data_frame)

with tab2:
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6,6))
        ax1 = conf_mat_fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(y_test, prediction, ax=ax1, normalize=True)
        st.pyplot(conf_mat_fig, use_container_width=True)
    st.divider()
    st.header("Classification Report")
    st.code(classification_report(y_test, prediction))

with tab3:

    sliders = []
    options = {}
    col1, col2 = st.columns(2)
    with col1:
        for feature_name in data_frame.columns:
            if feature_name not in ['cholesterol', 'gluc', 'diabetic']:
                slider_value = st.slider(label=feature_name, min_value=float(data_frame[feature_name].min()),
                                         max_value=float(data_frame[feature_name].max()))
                sliders.append(slider_value)
            else:
                # Use select box for features 'cholesterol', 'gluc', and 'diabetic'
                option = st.selectbox(label=feature_name, options=[1, 2, 3])  # Assuming options are 1, 2, and 3
                sliders.append(option)
                options[feature_name] = option

        input_data = pd.DataFrame([sliders], columns=data_frame.columns)
        input_data = calculate_and_add_bmi(input_data)
        print(input_data.to_string())
        #scaling
        selected_features = ['age', 'height', 'weight', 'bp_high', 'bp_lo', 'bmi']
        min_vals = filled_x_train[selected_features].min()
        max_vals = filled_x_train[selected_features].max()


        def min_max_scaler(row, min_vals, max_vals):
            scaled_row = (row[selected_features] - min_vals) / (max_vals - min_vals)
            return scaled_row

        scaled_input_values = min_max_scaler(input_data, min_vals, max_vals)

        input_data[selected_features] = scaled_input_values
        input_data = preprocess_input_data(input_data, filled_x_train)
        input_data = encode_input_data(input_data, options)
        print(input_data.to_string())
        st.write(input_data)
        y = samefeature(X_train, input_data)
        st.write(y)

    with col2:
        col1, col2 = st.columns(2, gap="medium")
        prediction = rf_classif.predict(y)
        predicted_class_label = 1 if prediction[0] == 1 else 0  # Adjust the prediction label based on the threshold
        with col1:
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(
                predicted_class_label), unsafe_allow_html=True)

            probs = rf_classif.predict_proba(y)
            probability = probs[0][prediction[0]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability * 100),
                      delta="{:.2f} %".format((probability - 0.5) * 100))

