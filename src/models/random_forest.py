from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

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