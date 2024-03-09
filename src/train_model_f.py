from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score


def svm_model_f(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='linear', C=1.0)
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("F1 Score:", f1)
    return accuracy
