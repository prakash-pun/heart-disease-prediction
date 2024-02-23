import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def alco_correlation():
    current_dir = os.getcwd()
    file_path = os.path.join("D:\\College\\Assignments\\Steps\\cardiovascular-disease-prediction\\data\\cleaned_data.xlsx")

    df = pd.read_excel(file_path)
    df_cp = df.copy()

    correlation_matrix = df_cp[['alco', 'cardio']].corr()

    
