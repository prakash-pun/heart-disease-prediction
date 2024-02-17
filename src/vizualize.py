import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_counts(col):
    levels, counts = np.unique(col, return_counts=True)
    return levels,counts

def barplot(levels,counts,col):
    plt.bar(levels, counts, color='skyblue')
    plt.xlabel('Levels')
    plt.ylabel('Frequency')
    plt.title(f'Frequency Count of {col} Levels')
    plt.show()
    
def histplot(col):
    plt.figure(figsize=(8,8))
    sns.displot(col,bins='auto')
    plt.show()
    
def scatter(data_frame,col):
    plt.figure(figsize=(8,8))
    sns.scatterplot(data_frame,x=data_frame.col,y=data_frame.cardio)
    plt.show()

def heatmap(data):
    plt.figure(figsize=(15,15))
    sns.heatmap(data, annot=True)
    plt.show()