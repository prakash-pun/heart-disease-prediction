"""Untitled1.py
Original file is located at
    https://colab.research.google.com/drive/114EwM4U9MTEpmfYlOUqvKQnr3s-u5-C2
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('merged_train_data.csv')
df

df['gender'].mean()

df['gender'].median()

df['gender'].std()

df['gender'].value_counts()

df.hist(column='gender', grid=False, edgecolor='black', color='red')

"""There are more Females than Males."""

df['bp_high'].mean()

df['bp_high'].median()

df['bp_high'].std()

df['bp_high'].value_counts()


df.hist(column='bp_high', grid=False, edgecolor='black', color = 'darkblue')

df['bp_lo'].describe()

df.hist(column='bp_lo', grid=False, edgecolor='black', color='purple')

df.hist(column='cholesterol_1', grid=False, edgecolor='black', color='blue')

df.hist(column='cholesterol_2', grid=False, edgecolor='blue', color='black')

df.hist(column='cholesterol_3', grid=False, edgecolor='black', color='red')

df.hist(column='gluc_1', grid=False, edgecolor='black', color='blue')

df.hist(column='gluc_2', grid=False, edgecolor='black', color='blue')

df.hist(column='gluc_3', grid=False, edgecolor='black', color='blue')

df.hist(column='diabetic_1', grid=False, edgecolor='black', color='red')

df.hist(column='diabetic_2', grid=False, edgecolor='black', color='red')

df.hist(column='diabetic_3', grid=False, edgecolor='black', color='red')
