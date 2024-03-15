from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb


def metrics(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions)

    return accuracy, precision, recall, f_score


class TrainModel():

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def svm_model(self):
        # Train the classifier
        svm_clf = svm.SVC(kernel='linear', C=1.0)
        svm_clf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = svm_clf.predict(self.X_test)
        result = metrics(self.y_test, predictions)

        return result

    def logistic_regression_model(self):
        # Train the classifier
        logreg = LogisticRegression(random_state=0)
        logreg.fit(self.X_train, self.y_train)

        # make prediction
        prediction = logreg.predict(self.X_test)
        result = metrics(self.y_test, prediction)

        return result

    def random_forest_model(self):
        # Train the classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=95)
        rf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = rf.predict(self.X_test)
        result = metrics(self.y_test, predictions)

        return result

    def xg_boost(self):
        # Define parameters for XGBoost
        params = {
            'booster': 'gbtree',
            'learning_rate': 0.15,
            'n_estimators': 200,
            'subsample': 0.8,
            'max_depth': 3,  # Tree Depth
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'merror'  # Evaluation metric
        }

        # XGB CLF
        xgb_clf = xgb.XGBClassifier(**params)
        xgb_clf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions_clf = xgb_clf.predict(self.X_test)
        binary_predictions_clf = (predictions_clf > 0.5).astype(int)

        # Calculate metrics
        result = metrics(self.y_test, binary_predictions_clf)

        return result

    def gbm_model(self):
        # Initialize the Gradient Boosting Classifier
        gradient_boosting = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        # Train the model
        gradient_boosting.fit(self.X_train, self.y_train)

        y_pred = gradient_boosting.predict(self.X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        return accuracy
