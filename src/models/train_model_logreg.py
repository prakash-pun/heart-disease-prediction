from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# logistic regression
def LogisticRegression_model1(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=0)
    # Train the classifier
    clf.fit(X_train, y_train)

    # make prediction
    prediction = clf.predict(X_test)

    # evaluate accuracy
    accuracy1 = accuracy_score(y_test, prediction)
    return accuracy1


def LogisticRegression_model2(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=0)
    # Train the classifier
    clf.fit(X_train, y_train)

    # make prediction
    prediction_score = clf.predict(X_test)

    # evaluate precision
    precision = precision_score(y_test, prediction_score)
    return precision