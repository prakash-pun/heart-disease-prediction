from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def random_forest_model(X_train, X_test, y_train, y_test):
    # Create a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Evaluate precision
    precision = precision_score(y_test, predictions)
    return precision


def random_forest_recall(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier(n_estimators=150, random_state=42)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = clf.predict(X_test)

    # Evaluate recall on the test data
    recall = recall_score(y_test, predictions)

