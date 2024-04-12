import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix
import streamlit as st
import io
from PIL import Image
def samefeature(x, y):
    training_feature_columns = x.columns

    # Get the list of feature columns present in the input data
    input_feature_columns = y.columns

    # Check if there are any columns in the input data that are not present in the training data
    extra_columns_in_input = set(
        input_feature_columns) - set(training_feature_columns)
    if extra_columns_in_input:
        y = y.drop(extra_columns_in_input, axis=1)
        print("Dropped extra columns from input data:", extra_columns_in_input)
    else:
        print("Input data has the same features as training data.")

    return y


def calculate_and_add_bmi(input_data):
    # Calculate BMI
    input_data['bmi'] = input_data['weight'] / (input_data['height'] ** 2)
    input_data['bmi'] = input_data['bmi'].round()

    return input_data


def preprocess_input_data(input_data, training_data):
    # Ensure input data contains all one-hot encoded features with initial values set to 0
    for col in training_data.columns:
        if col.startswith("cholesterol_") or col.startswith("gluc_") or col.startswith("diabetic_"):
            if col not in input_data.columns:
                input_data[col] = 0

    return input_data


def encode_input_data(input_data, user_input):
    # Update input data with user input values
    for feature, value in user_input.items():
        if feature in input_data.columns:
            if feature in ['cholesterol', 'gluc', 'diabetic']:
                # Update the corresponding one-hot encoded feature to have a value of 1
                feature_prefix = feature + '_' + str(value)
                if feature_prefix in input_data.columns:
                    input_data[feature_prefix] = 1
            else:
                input_data[feature] = value  # Update other features as usual

    return input_data

def plot_roc_curve(fpr, tpr):
    plt.figure()  # Create a new figure
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Display the plot image with specified dimensions
    st.image(buf, width=800, caption='ROC Curve')


def calculate_metrics(y_true, y_pred, metric):
    if metric == 'F1 Score':
        score = f1_score(y_true, y_pred, average='weighted')
    elif metric == 'Precision':
        score = precision_score(y_true, y_pred, average='weighted')
    elif metric == 'Accuracy':
        score = accuracy_score(y_true, y_pred)
    elif metric == 'Recall':
        score = recall_score(y_true, y_pred, average='weighted')
    else:
        score = None

    return score

def plot_confusion_matrix(y_test, prediction):
    conf_mat_fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figsize here
    conf_matrix = confusion_matrix(y_test, prediction)
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.set_title('Confusion Matrix (XGBoost)')

    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center")

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()

    # Create colorbar
    plt.colorbar(im, ax=ax)

    # Convert Matplotlib figure to PIL image
    buf = io.BytesIO()
    conf_mat_fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)

    # Display image in Streamlit with specified width
    st.image(image, width=800)
