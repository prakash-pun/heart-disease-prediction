from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def Swetha_KNNmodel(X_train, X_test, y_train, y_test, k=10):

    # Training the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return knn_classifier, accuracy, precision, recall, f1, roc_auc


def Swetha_DecisionTreeModel(X_train, X_test, y_train, y_test):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        # Maximum depth of the tree
        'max_depth': [None, 5, 10, 15],
        # Minimum number of samples required to split an internal node
        'min_samples_split': [2, 5, 10],
        # Minimum number of samples required to be at a leaf node
        'min_samples_leaf': [1, 2, 4]
    }

    # Create the Decision Tree classifier
    Dec_Tree_classifier = DecisionTreeClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=Dec_Tree_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the hyperparameters
    param = grid_search.best_params_
    print("Hyperparameters:", param)

    # Use the best hyperparameters to train the final model
    best_max_depth = param['max_depth']
    best_min_samples_split = param['min_samples_split']
    best_min_samples_leaf = param['min_samples_leaf']

    # Train the Decision Tree classifier with the hyperparameters
    dt_param = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf)
    dt_param.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = dt_param.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return dt_param, accuracy, precision, recall, f1, roc_auc
