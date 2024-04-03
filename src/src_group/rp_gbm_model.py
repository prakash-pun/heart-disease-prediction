from sklearn.ensemble import GradientBoostingClassifier
#from .qhull import *
#import qhull
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

print("hii")
def metrics(y_test, predictions, proba):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, proba)

    return accuracy, precision, recall, f_score, roc_auc



def rp_gbm_model(X_train, X_test, y_train, y_test):
    # Initialize the Gradient Boosting Classifier
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=180, learning_rate=0.12, max_depth=2)

    # Train the model
    gradient_boosting.fit(X_train, y_train)

    # Predictions
    train_predict = gradient_boosting.predict(X_train)
    train_proba = gradient_boosting.predict_proba(X_train)

    prediction = gradient_boosting.predict(X_test)
    predict_proba = gradient_boosting.predict_proba(X_test)

    # Calculate metrics
    rst = metrics(y_test, prediction, predict_proba[:, 1])
    rst_train = metrics(y_train, train_predict, train_proba[:, 1])

    return rst_train, rst