from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def random_forest_model(X_train, X_test, y_train, y_test):
    #Train the classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=80, max_depth=None, min_samples_split=4, min_samples_leaf=1)
        rf.fit(X_train, y_train)

        # Make predictions
        predictions = rf.predict(X_test)
        print(predictions)


        # Evaluate metrics
        precision = precision_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f_score= f1_score(y_test, predictions)

        return accuracy, precision, recall, f_score