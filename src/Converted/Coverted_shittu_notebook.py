import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C://Users//USER//Documents//shittu_data.csv')

df.head()

df.shape

df.isnull().sum()

def derive_diabetic(row):
    if row['cholesterol'] == 3 or row['gluc'] == 3:
        return 3  # Diabetic
    elif row['gluc'] == 2:
        return 2  # Prediabetic
    else:
        return 1  # Non-diabetic

# Apply the function to derive diabetic status
df['diabetic'] = df.apply(derive_diabetic, axis=1)

df.head()

df.columns

newdata =df[['cholesterol','gluc', 'diabetic']]

newdata.corr()

sns.heatmap(newdata.corr(),annot=True)
plt.show()

df.to_csv('shittu_datanew.csv', index= False)