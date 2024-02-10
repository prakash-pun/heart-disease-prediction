import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Data Import
df2=pd.read_csv("train_data.csv",index_col=0,na_values="??")
df=df2.copy(deep=True)

head=df.head(10)
tail=df.tail(10)
uniques_vals=df.nunique()

# Basic Stats
null=df.isnull().sum()
stats=df.describe()
df.info()
corr=df.corr(method='pearson')
df['cholesterol'].corr(df['cardio'])
df['gluc'].corr(df['cardio'])

# % Plotting
npchk=df[['cholesterol','gluc','diabetic','cardio']]
nplist=npchk.columns.tolist()

def get_counts(col):
    levels, counts = np.unique(col, return_counts=True)
    return levels,counts

def barplt(levels,counts,col):
    plt.bar(levels, counts, color='skyblue')
    plt.xlabel('Levels')
    plt.ylabel('Frequency')
    plt.title(f'Frequency Count of {col} Levels')
    plt.show()

#Distribution for each column
for i in nplist:
    levels, counts = get_counts(df[i])
    barplt(levels,counts,i)

# % Encoding needs tuning
encoder = OneHotEncoder()

ohreq=df[['cholesterol','gluc','diabetic']]
oh_encoded=encoder.fit_transform(ohreq)
oh_df = pd.DataFrame(oh_encoded.toarray(), columns=encoder.get_feature_names_out(['cholesterol', 'gluc', 'diabetic']))
oh_df.index=df.index
df=pd.concat([df,oh_df],axis=1)

# % Export Data
df.to_csv("aman_train_data.csv")