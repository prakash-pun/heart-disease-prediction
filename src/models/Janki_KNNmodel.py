from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def KNN_model(X_train, X_test, y_train, y_test):
    # Accuracy
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    # Make predictions
    y_pred = knn_model.predict(X_test)

    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    
    # Precision
    precision = precision_score(y_test, y_pred, average='weighted')
    
    # Recall
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)  

return accuracy, precision, recall, f1, cm
    




