import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_data  

data_frame = get_data("cleaned_data.csv")

def alco_correlation(data_frame):
    df_cp = data_frame.copy()

    correlation_matrix = df_cp[['alco', 'cardio']].corr()

    sns.histplot(df_cp['alco'], bins=1, kde=True)
    plt.title('Weight Distribution')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Alcohol and Cardio')
    plt.show()


def weight_correlation(data_frame):

    df1=data_frame.copy(deep=True)
    usedf=df1[['weight','cardio']].copy()

    usedf.isna().sum()
    usedf.dropna(axis=0,inplace=True)

    w200=usedf.loc[(usedf.weight>=100) & (usedf.weight<=200)]
    uppercor=w200.corr(method='pearson')

    w100=usedf.loc[(usedf.weight>=71) & (usedf.weight<=99)]
    midcor=w100.corr(method='pearson')

    w70=usedf.loc[(usedf.weight>=30) & (usedf.weight<=70)]
    lowcor=w70.corr(method='pearson')


def smoke_correlation(data_frame):
    # Extract the 'smoke', 'alco', and 'cardio' columns
    selected_columns = data_frame[['smoke', 'alco', 'cardio']]

    # Display the correlation matrix
    correlation_matrix = selected_columns.corr()

    # Plot a simple heatmap to visualize the correlation
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Heatmap: Smoke, Alcohol, and Cardio')
    plt.show()


# physical_activity_corr_cardio_py(3).py
def active_correlation(data_frame):

    my_column = data_frame[['active','cardio']]

    sns.histplot(data_frame['active'], bins = 20 , kde = True)
    plt.title('Active Distribution')
    plt.xlabel('active')
    plt.ylabel('Frequency')
    plt.show()

    data_frame.plot(kind='scatter', x='active', y='cardio')
    plt.gca().spines[['top', 'right',]].set_visible(False)

    correlation_matrix = data_frame[['active','cardio']].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation matrix of active and cardio')
    plt.show()


def cholestrol_corr(data_frame):
    # Extract the 'weight', 'cholesterol', and 'cardio' columns
    selected_columns = data_frame[['weight', 'cholesterol', 'cardio']]

    # Display the correlation matrix
    correlation_matrix = selected_columns.corr()

    # Plot a heatmap to visualize the correlation
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Heatmap: Cholesterol, Weight, and Cardio')
    plt.show()

