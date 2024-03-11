from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score

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

def random_forest_model2(X_train, X_test, y_train, y_test):
    # Create a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Evaluate accuracy and f1 score
    accuracy = accuracy_score(y_test, predictions)
    f_score = f1_score(y_test, predictions, average="binary")

    return accuracy, f_score


