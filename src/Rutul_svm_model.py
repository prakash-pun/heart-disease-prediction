
from sklearn.metrics import accuracy_score, precision_score ,f1_score, recall_score
from sklearn import svm

def rp_svm_model(X_train, X_test, y_train, y_test):
    prp = svm.SVC(kernel='linear')
    prp.fit(X_train,y_train)
    y_train_pred = prp.predict(X_train)
    y_test_pred = prp.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    train_f1_score = f1_score(y_train, y_train_pred)
    test_f1_score = f1_score(y_test, y_test_pred)
    train_recall_score = recall_score(y_train, y_train_pred)
    test_recall_score = recall_score(y_test, y_test_pred)


    return train_accuracy, test_accuracy , train_precision , test_precision , train_f1_score, test_f1_score , train_recall_score, test_recall_score


