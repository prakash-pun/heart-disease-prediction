import pandas as pd
df=pd.read_csv("aman_py_data.csv",index_col=0,na_values=["??","#","???"])

df1=df.copy(deep=True)
stats=df1.describe()
usedf=df1[['weight','cardio']].copy()

statsuse=usedf.describe()
usedf.isna().sum()
usedf.dropna(axis=0,inplace=True)
head=usedf.head(10)
tail=usedf.tail()

w200=usedf.loc[(usedf.weight>=100) & (usedf.weight<=200)]
uppercor=w200.corr(method='pearson')

w100=usedf.loc[(usedf.weight>=71) & (usedf.weight<=99)]
midcor=w100.corr(method='pearson')

w70=usedf.loc[(usedf.weight>=30) & (usedf.weight<=70)]
lowcor=w70.corr(method='pearson')


