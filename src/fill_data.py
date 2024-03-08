import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import get_data

# df2=pd.read_csv(get_csv_file("train_data.csv"),index_col=0,na_values="??")
data = get_data("test_data.csv")


def fill_data(data_frame, file_name="merged_test_data.csv"):
    """Fill the missing value

    Parameters
    ----------
    data_frame : DataFrame
        data frame that you want to fill
    file_name : string, optional
        Name of the new filled dataset

    Returns
    --------
    new filled data frame
    """

    df = data_frame.copy(deep=True)
    # Basic Stats
    m_bp_lo = df.loc[:, 'bp_lo'].mean()
    m_round = round(m_bp_lo, -1)
    df['bp_lo'].fillna(value=m_round, inplace=True)
    df['height_m'] = df['height'] / 100

    # Calculate BMI
    df['bmi'] = df['weight'] / (df['height_m'] ** 2)
    # Round BMI to the nearest whole number
    df['bmi'] = df['bmi'].round()

    df['cholesterol'].corr(df['cardio'])
    df['gluc'].corr(df['cardio'])

    # % Encoding needs tuning
    encoder = OneHotEncoder()

    ohreq = df[['cholesterol', 'gluc', 'diabetic']]
    oh_encoded = encoder.fit_transform(ohreq)
    oh_df = pd.DataFrame(oh_encoded.toarray(), columns=encoder.get_feature_names_out(
        ['cholesterol', 'gluc', 'diabetic']))
    oh_df.index = df.index
    df = pd.concat([df, oh_df], axis=1)

    # % Export Data
    df.to_csv(f"data/{file_name}")

    return df


fill_data(data)
