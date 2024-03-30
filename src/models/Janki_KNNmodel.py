import numpy as np
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def plot_roc(fpr, tpr):
    # Plot ROC curve
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='red',
             linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def metrics(y_test, predictions, proba):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, proba)

    return accuracy, precision, recall, f_score, roc_auc


class TrainModel():

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def logistic_regression_model(self):
        # Train the classifier
        logreg = LogisticRegression(random_state=42,max_iter=1000,test_size=0.2)
        logreg.fit(self.X_train, self.y_train)

      # Define the hyperparameter space for the random search
param_distributions = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    
}  

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=10000, random_state=42)

 # Set up the random search with cross-validation
 random_search = RandomizedSearchCV(logreg, param_distributions=param_distributions, n_iter=50, cv=5, random_state=42, n_jobs=-1)

# Fit the random search model
random_search.fit(X_train_scaled, y_train)

# Evaluate the best model
best_model = random_search.best_estimator_
predictions = best_model.predict(X_test_scaled)
proba = best_model.predict_proba(X_test_scaled)[:, 1]

# make prediction
train_predict = logreg.predict(self.X_train)
train_proba = logreg.predict_proba(self.X_train)

prediction = logreg.predict(self.X_test)
test_proba = logreg.predict_proba(self.X_test)

result_test = metrics(self.y_test, prediction, test_proba[:, 1])
result_train = metrics(self.y_train, train_predict, train_proba[:, 1])

        return result_train, result_test

   

    