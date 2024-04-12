import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import streamlit as st
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from lime import lime_tabular
from sklearn.preprocessing import MinMaxScaler
from models.dash import samefeature, calculate_and_add_bmi, preprocess_input_data, encode_input_data, plot_roc_curve, calculate_metrics
from models.feature_importance_analysis import FeatureImportanceAnalysis
from feature_engineering import FeatureEngines
from data_preprocessing import DataProcessor
from utils import DataInitializer


# Set the page config
st.set_page_config(page_title='Heart Diesease Prediction',
                   layout='wide',
                   page_icon='♥️', initial_sidebar_state="expanded")

st.title(
    "Heart Disease :red[Prediction] :bar_chart: :chart_with_upwards_trend:")

# Define custom CSS to reduce top padding
custom_css = """
<style>
.st-emotion-cache-z5fcl4 {
    padding-top: 2rem !important; /* Adjust the value as needed */
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    ["Data :clipboard:", "Performance :weight_lifter:", "Prediction :bicyclist:"])


di = DataInitializer()
dp = DataProcessor()
fe = FeatureEngines()

data_frame = di.get_data()

X_train, X_test, y_train, y_test = di.split_data()
X_train1, X_test1, y_train1, y_test1 = di.split_data()

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
# prediction = rf_classif.predict(X_test)
probabilities = rf_classif.predict_proba(X_test)
prediction = probabilities.argmax(axis=1)


# Instantiate the FeatureImportanceAnalysis class
feature_analysis = FeatureImportanceAnalysis(
    model_files={"XGBoost": "xg_boost_model.joblib"}, X_test=X_test, y_test=y_test)


with tab1:
    st.header("📊  Data Visualizer")
    col1, col2 = st.columns([1, 2])
    col3, col4 = st.columns([2, 1])

    columns = data_frame.columns.tolist()

    with col1:
        st.subheader("Total Dataset")
        st.markdown(f"Total rows: {data_frame.shape[0]}")
        attribute = st.selectbox("Select Attribute", [
            "Gender", "Age", "Cholesterol", "Weight", "Diabetic", "Systolic Blood Pressure (bp_high)", "Smoking", "Activity"])

    with col2:
        def create_bar_chart():
            # Group data by gender and count occurrences
            gender_counts = data_frame['gender'].map(
                {1: 'Male', 2: 'Female'}).value_counts()

            # Create bar chart
            fig = px.bar(gender_counts, x=gender_counts.index,
                         y=gender_counts.values, labels={'x': 'Gender', 'y': 'Count'})

            # Update layout
            fig.update_layout(title="Gender Distribution")

            return fig

        # Function to create interactive pie chart
        def create_pie_chart():
            # Group data by age group and count occurrences
            age_bins = [30, 40, 50, 60, 70, 80, 90, 100]
            age_labels = ['30-39', '40-49', '50-59',
                          '60-69', '70-79', '80-89', '90-100']
            data_frame['age_group'] = pd.cut(
                data_frame['age'], bins=age_bins, labels=age_labels, right=False)
            age_counts = data_frame['age_group'].value_counts()

            # Create pie chart
            fig = px.pie(age_counts, values=age_counts.values,
                         names=age_counts.index, title="Age Distribution")

            return fig

        def create_chart(df):
            if attribute == "Cholesterol":
                # Group data by cholesterol level and count occurrences
                cholesterol_counts = df['cholesterol'].value_counts()

                # Create chart
                title = "Cholesterol Distribution"
                fig = px.bar(x=cholesterol_counts.index, y=cholesterol_counts.values, labels={
                             'x': 'Cholesterol Level', 'y': 'Count'}, title=title)
            elif attribute == "Weight":
                # Create chart for weight distribution
                fig = px.histogram(df, x='weight', title="Weight Distribution")
            elif attribute == "Diabetic":
                # Group data by diabetic status and count occurrences
                diabetic_counts = df['diabetic'].value_counts()

                # Create chart
                title = "Diabetic Status Distribution"
                fig = px.bar(x=diabetic_counts.index, y=diabetic_counts.values, labels={
                             'x': 'Diabetic Status', 'y': 'Count'}, title=title)
            elif attribute == "Systolic Blood Pressure (bp_high)":
                # Create chart for bp_high distribution
                fig = px.histogram(
                    df, x='bp_high', title="Systolic Blood Pressure (bp_high) Distribution")
            elif attribute == "Smoking":
                # Group data by smoking status and count occurrences
                smoking_counts = df['smoke'].value_counts()

                # Create chart
                title = "Smoking Status Distribution"
                fig = px.bar(x=smoking_counts.index, y=smoking_counts.values, labels={
                             'x': 'Smoking Status', 'y': 'Count'}, title=title)
            elif attribute == "Activity":
                # Group data by activity level and count occurrences
                activity_counts = df['active'].map(
                    {0: 'Inactive', 1: 'Active'}).value_counts()

                # Create chart
                title = "Activity Level Distribution"
                fig = px.bar(x=activity_counts.index, y=activity_counts.values, labels={
                             'x': 'Activity Level', 'y': 'Count'}, title=title)

            return fig

        # Display chart
        if attribute == "Gender":
            st.plotly_chart(create_bar_chart())
        elif attribute == "Age":
            st.plotly_chart(create_pie_chart())
        else:
            st.plotly_chart(create_chart(data_frame))

    with col3:
        st.write(data_frame)

    with col4:
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
        conf_mat_fig = plt.figure(figsize=(10, 10))
        ax1 = conf_mat_fig.add_subplot(222)
        conf_matrix = confusion_matrix(
            y_test, prediction)
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
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

    # Get available metrics
    available_metrics = ['F1 Score', 'Precision', 'Accuracy', 'Recall']

    # Allow user to select metric
    selected_metric = st.selectbox('Select Metric', available_metrics)

    # Calculate selected metric
    metric_score = calculate_metrics(y_test, prediction, selected_metric)

    # Display metric score
    st.header("Evaluation Metric")
    st.write(f"{selected_metric}: {metric_score}")

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])

    # Display ROC curve
    st.header("ROC Curve")
    st.write("False Positive Rate vs True Positive Rate")
    plot_roc_curve(fpr, tpr)

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
    # input data
    sliders = []
    options = {}
    alias_names = {
        'age': 'Age',
        'gender': 'Gender',
        'height': 'Height',
        'weight': 'Weight',
        'smoke': 'Smoker',
        'alco': 'Alcohol Consumer',
        'active': 'Excercise',
        'bp_high': 'High Blood Pressure',
        'bp_lo': 'Low Blood Pressure'
    }
    for feature_name in X_train1.columns:
        if feature_name not in ['cholesterol', 'gluc', 'diabetic']:
            if feature_name in ['gender', 'smoke', 'alco', 'active']:
                option = st.selectbox(label=alias_names.get(
                    feature_name, feature_name), options=[0, 1])
                sliders.append(option)
                options[feature_name] = option
            else:
                slider_value = st.slider(label=alias_names.get(feature_name, feature_name),
                                         min_value=float(
                                             X_train1[feature_name].min()),
                                         max_value=float(X_train1[feature_name].max()))
                sliders.append(slider_value)
        else:
            option = st.selectbox(label=alias_names.get(
                feature_name, feature_name), options=[1, 2, 3])
            sliders.append(option)
            options[feature_name] = option

    input_data = pd.DataFrame([sliders], columns=X_train1.columns)
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
    # st.write(input_data)
    y = samefeature(X_train, input_data)
    # st.write(y)

    # prediction
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
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=X_train.columns)
    exp = explainer.explain_instance(
        y.values[0], rf_classif.predict_proba, num_features=len(X_train.columns))

    # Display the explanation
    st.sidebar.subheader("Local Explanation")
    st.sidebar.write(exp.as_list())

    # Visualize explanation in main section
    st.subheader("Local Explanation Visualization")
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
