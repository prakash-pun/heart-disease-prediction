from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

def svm_model_training(X_train, X_test, y_train, y_test, kernel='linear'):
    svm_model_tirth = SVC(kernel=kernel)

    # Training the svm model
    svm_model_tirth.fit(X_train, y_train)

    # Making predictions on the testing set
    y_pred = svm_model_tirth.predict(X_test)

    # Calculating accuracy and precision
    svm_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return svm_accuracy, precision