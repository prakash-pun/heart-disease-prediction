import pandas as pd
def samefeature(x,y):
    training_feature_columns = x.columns

    # Get the list of feature columns present in the input data
    input_feature_columns = y.columns

    # Check if there are any columns in the input data that are not present in the training data
    extra_columns_in_input = set(input_feature_columns) - set(training_feature_columns)
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