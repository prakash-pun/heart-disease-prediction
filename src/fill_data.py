import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def fill_data(data_frame):
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
    # df['bp_lo'].fillna(value=m_round, inplace=True)
    df.fillna({"bp_lo": m_round}, inplace=True)
    df['height_m'] = df['height'] / 100

    # Calculate BMI
    df['bmi'] = df['weight'] / (df['height_m'] ** 2)
    df['bmi'] = df['bmi'].round()

    # % Encoding needs tuning
    encoder = OneHotEncoder()

    ohreq = df[['cholesterol', 'gluc', 'diabetic']]
    oh_encoded = encoder.fit_transform(ohreq)
    oh_df = pd.DataFrame(oh_encoded.toarray(), columns=encoder.get_feature_names_out(
        ['cholesterol', 'gluc', 'diabetic']))
    oh_df.index = df.index
    df = pd.concat([df, oh_df], axis=1)

    return df
