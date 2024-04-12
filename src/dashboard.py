import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from models.dash import samefeature, calculate_and_add_bmi, preprocess_input_data, encode_input_data
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import FeatureEngines
from data_preprocessing import DataProcessor
from models.feature_importance_analysis import FeatureImportanceAnalysis
from utils import DataInitializer
# import scikitplot as skplt
from lime import lime_tabular


# Set the page config
st.set_page_config(page_title='Heart Diesease Prediction',
                   layout='centered',
                   page_icon='♥️')

st.title(
    "Heart Disease :red[Prediction] :bar_chart: :chart_with_upwards_trend:")
st.markdown("Predict heart Type using different parameters")

tab1, tab2, tab3 = st.tabs(
    ["Data :clipboard:", "Performance :weight_lifter:", "Local Performance :bicyclist:"])


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


working_dir = os.path.dirname(os.path.abspath(__file__))


rf_classif = load(f"{working_dir}/dump_model/xg_boost_model.joblib")

# for checking if this is working or not
prediction = rf_classif.predict(X_test)

working_dir = os.path.dirname(os.path.abspath(__file__))
rf_classif = load(f"{working_dir}/dump_model/xg_boost_model.joblib")

# Instantiate the FeatureImportanceAnalysis class
feature_analysis = FeatureImportanceAnalysis(model_files={"XGBoost": "xg_boost_model.joblib"}, X_test=X_test, y_test=y_test)

with tab1:
    st.header("📊  Data Visualizer")
    col1, col2 = st.columns(2)
    columns = data_frame.columns.tolist()

    with col1:
        st.write(data_frame)

    with col2:
        # Allow the user to select columns for plotting
        x_axis = st.selectbox('Select the X-axis', options=columns+["None"])
        y_axis = st.selectbox('Select the Y-axis', options=columns+["None"])

        plot_list = ['Line Plot', 'Bar Chart',
                     'Scatter Plot', 'Distribution Plot', ]
        # Allow the user to select the type of plot
        plot_type = st.selectbox('Select the type of plot', options=plot_list)

    # Generate the plot based on user selection
    if st.button('Generate Plot'):

        fig, ax = plt.subplots(figsize=(6, 4))

        if plot_type == 'Line Plot':
            sns.lineplot(x=data_frame[x_axis], y=data_frame[y_axis], ax=ax)
        elif plot_type == 'Bar Chart':
            sns.barplot(x=data_frame[x_axis], y=data_frame[y_axis], ax=ax)
        elif plot_type == 'Scatter Plot':
            sns.scatterplot(x=data_frame[x_axis], y=data_frame[y_axis], ax=ax)
        elif plot_type == 'Distribution Plot':
            sns.histplot(data_frame[x_axis], kde=True, ax=ax)
            y_axis = 'Density'

        # Adjust label sizes
        ax.tick_params(axis='x', labelsize=10)  # Adjust x-axis label size
        ax.tick_params(axis='y', labelsize=10)  # Adjust y-axis label size

        # Adjust title and axis labels with a smaller font size
        plt.title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize=12)
        plt.xlabel(x_axis, fontsize=10)
        plt.ylabel(y_axis, fontsize=10)

        # Show the results
        st.pyplot(fig)


with tab2:
    st.header("Confusion Matrix ")
    col1, col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6, 6))
        ax1 = conf_mat_fig.add_subplot(111)
        conf_matrix = confusion_matrix(
            y_test, prediction)
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (XGBoost)')
        plt.colorbar()

        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(conf_matrix[i, j]),
                         horizontalalignment="center")

        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.show()
        st.pyplot(conf_mat_fig, use_container_width=True)
    st.divider()
    st.header("Classification Report")
    st.code(classification_report(y_test, prediction))

    st.header("Model Performance Metrics")

    # Plot Feature Importance
    feature_importance_plots = feature_analysis.plot_feature_importance()
    for model_name, plot_path in feature_importance_plots.items():
        st.image(plot_path, caption=f"Feature Importance - {model_name}")

    # Permutation Importance Analysis
    permutation_importance_plots = feature_analysis.permutation_importance_analysis()
    for model_name, plot_path in permutation_importance_plots.items():
        st.image(plot_path, caption=f"Permutation Importance - {model_name}")

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
                # Assuming options are 1, 2, and 3
                option = st.selectbox(label=feature_name, options=[1, 2, 3])
                sliders.append(option)
                options[feature_name] = option

        input_data = pd.DataFrame([sliders], columns=data_frame.columns)
        input_data = calculate_and_add_bmi(input_data)
        print(input_data.to_string())
        # scaling
        selected_features = ['age', 'height',
                             'weight', 'bp_high', 'bp_lo', 'bmi']
        min_vals = filled_x_train[selected_features].min()
        max_vals = filled_x_train[selected_features].max()

        def min_max_scaler(row, min_vals, max_vals):
            scaled_row = (row[selected_features] -
                          min_vals) / (max_vals - min_vals)
            return scaled_row
        # scaling input values
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
        # Adjust the prediction label based on the threshold
        predicted_class_label = 1 if prediction[0] == 1 else 0
        with col1:
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(
                predicted_class_label), unsafe_allow_html=True)

            probs = rf_classif.predict_proba(y)
            probability = probs[0][prediction[0]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability * 100),
                      delta="{:.2f} %".format((probability - 0.5) * 100))

        # Explanation Section in sidebar
        st.sidebar.title("Explanation")
        # LIME Explanation
        explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns)
        exp = explainer.explain_instance(y.values[0], rf_classif.predict_proba, num_features=len(X_train.columns))

        # Display the explanation
        st.sidebar.subheader("Local Explanation")
        st.sidebar.write(exp.as_list())

        # Visualize explanation in main section
        st.subheader("Local Explanation Visualization")
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)


