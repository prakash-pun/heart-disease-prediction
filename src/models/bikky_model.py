from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt


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
        logreg = LogisticRegression(random_state=0,max_iter=1000)
        logreg.fit(self.X_train, self.y_train)

        # make prediction
        train_predict = logreg.predict(self.X_train)
        train_proba = logreg.predict_proba(self.X_train)

        prediction = logreg.predict(self.X_test)
        test_proba = logreg.predict_proba(self.X_test)

        result_test = metrics(self.y_test, prediction, test_proba[:, 1])
        result_train = metrics(self.y_train, train_predict, train_proba[:, 1])

        return result_train, result_test

    def xg_boost(self):
        # Define parameters for XGBoost
        params = {
            'booster': 'gbtree',  # gblinear
            'learning_rate': 0.9,
            'n_estimators': 200,
            'subsample': 0.8,
            'max_depth': 3,  # Tree Depth
            'objective': 'binary:logistic', # Binary classification
            'eval_metric': 'merror'  # Evaluation metric
        }
    
        # XGB CLF
        xgb_clf = xgb.XGBClassifier(**params)
        xgb_clf.fit(self.X_train, self.y_train)

        # Make predictions
        train_predict = xgb_clf.predict(self.X_train)
        train_predictions_clf = (train_predict > 0.5).astype(int)
        train_proba = xgb_clf.predict_proba(self.X_train)

        predictions_clf = xgb_clf.predict(self.X_test)
        binary_predictions_clf = (predictions_clf > 0.5).astype(int)
        predict_proba = xgb_clf.predict_proba(self.X_test)

        # Calculate metrics
        result = metrics(self.y_test, binary_predictions_clf,
                         predict_proba[:, 1])
        result_train = metrics(
            self.y_train, train_predictions_clf, train_proba[:, 1])

        return result_train, result

    def gbm_model(self):
        # Initialize the Gradient Boosting Classifier
        gradient_boosting = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.14, max_depth=2)

        # Train the model
        gradient_boosting.fit(self.X_train, self.y_train)

        # Predictions
        train_predict = gradient_boosting.predict(self.X_train)
        train_proba = gradient_boosting.predict_proba(self.X_train)

        prediction = gradient_boosting.predict(self.X_test)
        predict_proba = gradient_boosting.predict_proba(self.X_test)

        # Calculate metrics
        result = metrics(self.y_test, prediction, predict_proba[:, 1])
        result_train = metrics(self.y_train, train_predict, train_proba[:, 1])

        return result_train, result