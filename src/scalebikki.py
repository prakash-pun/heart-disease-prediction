import pandas as pd
df = pd.read_csv("merged_train_data.csv", index_col=0)
df.head()
df.shape
df.dtypes
X = df.iloc[:, [0,12]]
Y = df.iloc[:, 12]
X.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 40)
X_train.head()
scaler = StandardScaler().fit(X_train)
print(scaler)
scaler.mean_
scaler.scale_
scaler.transform(X_train)
X_train_scaled = scaler.transform(X_train)
print (X_train_scaled)
print (X_train_scaled.mean(axis=0))
print (X_train_scaled.std(axis=0))
X_test.head()
scaler = StandardScaler().fit(X_test)
scaler.mean_
scaler.scale_
scaler.transform(X_test)
X_test_scaled = scaler.transform(X_test)
print (X_test_scaled)
print (X_test_scaled.mean(axis=0))
print (X_test_scaled.std(axis=0))
X_test.head()
scaler = StandardScaler().fit(X_test)
scaler.mean_
scaler.scale_
scaler.transform(X_test)
X_test_scaled = scaler.transform(X_test)
print (X_test_scaled)
print (X_test_scaled.mean(axis=0))
print (X_test_scaled.std(axis=0))