from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from sklearn.linear_model import LogisticRegression
from feature_extraction import extract_feature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from utils import get_data
import streamlit as st
from joblib import load, dump
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from models.dash import samefeature, calculate_and_add_bmi, preprocess_input_data, encode_input_data


data_frame = get_data("cleaned_data.csv")

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

# metrics
def metrics(y_test, predictions, proba):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, proba)

    return accuracy, precision, recall, f_score, roc_auc

# running logistic model
# Train the classifier
logreg = LogisticRegression(random_state=0,max_iter=1000)
logreg.fit(X_train, y_train)

# make prediction
train_predict = logreg.predict(X_train)
train_proba = logreg.predict_proba(X_train)

prediction = logreg.predict(X_test)
test_proba = logreg.predict_proba(X_test)

result_test = metrics(y_test, prediction, test_proba[:, 1])
result_train = metrics(y_train, train_predict, train_proba[:, 1])

print("Logistic Regression:", result_test,result_train)

# saving a model
dump(logreg,"../src/models/log_ms.model")

# load model
rf_classif = load("../src/models/log_ms.model")

# for checking if this is working or not
y=rf_classif.predict(X_test)

print(y)

st.title("heart disease Prediction :red[Prediction] :bar_chart: :chart_with_upwards_trend:")
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
        y=preprocess_input_data(input_data, filled_x_train)
        y=encode_input_data(y, options)

        y = samefeature(X_train, y)
        print(y.to_string())

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

