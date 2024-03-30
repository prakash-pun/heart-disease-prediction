from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


# import lightgbm as lgbm


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
        logreg = LogisticRegression(random_state=0, max_iter=1000)
        logreg.fit(self.X_train, self.y_train)

        # make prediction
        train_predict = logreg.predict(self.X_train)
        train_proba = logreg.predict_proba(self.X_train)

        prediction = logreg.predict(self.X_test)
        test_proba = logreg.predict_proba(self.X_test)

        result_test = metrics(self.y_test, prediction, test_proba[:, 1])
        result_train = metrics(self.y_train, train_predict, train_proba[:, 1])

        # Calculate feature importances
        feature_importance = np.abs(logreg.coef_[0])
        feature_importance /= feature_importance.sum()
        feature_names = self.X_train.columns.tolist()

        return result_train, result_test, feature_importance, feature_names

    def xg_boost(self):
        # Define parameters for XGBoost
        params = {
            'booster': ['gbtree', 'gblinear'],  # gblinear
            'learning_rate': np.arange(0.01, 0.9, 0.01),
            'n_estimators': range(50, 1000, 50),
            'objective': ['binary:logistic'],
            # ,'multi:softmax','multi:softprob','reg:logitstic'],  # Binary classification
            'eval_metric': ['merror', 'logloss', 'auc'],  # Evaluation metric
            # 'subsample': np.arange(0.1, 0.9, 0.1),
            # 'max_depth': range(2, 7)
        }
        model = xgb.XGBClassifier()
        random_search = RandomizedSearchCV(estimator=model, param_distributions=params,
                                           n_iter=100, scoring="recall", cv=5, random_state=42, n_jobs=-1)
        random_search.fit(self.X_train, self.y_train)

        best_param = random_search.best_params_

        # XGB CLF
        xgb_clf = xgb.XGBClassifier(**best_param)
        xgb_clf.fit(self.X_train, self.y_train)

        # Make predictions
        train_predict = xgb_clf.predict(self.X_train)
        train_predictions_clf = (train_predict > 0.5).astype(int)
        train_proba = xgb_clf.predict_proba(self.X_train)

        predictions_clf = xgb_clf.predict(self.X_test)
        binary_predictions_clf = (predictions_clf > 0.5).astype(int)
        predict_proba = xgb_clf.predict_proba(self.X_test)

        # Calculate metrics
        result_train = metrics(self.y_train, train_predictions_clf, train_proba[:, 1])
        result = metrics(self.y_test, binary_predictions_clf, predict_proba[:, 1])

        # Calculate feature importances
        feature_importance = xgb_clf.feature_importances_
        feature_names = self.X_train.columns.tolist()

        return result_train, result, feature_importance, feature_names

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

        # Calculate feature importances
        feature_importance = gradient_boosting.feature_importances_
        feature_names = self.X_train.columns.tolist()

        return result_train, result, feature_importance, feature_names

    # Unused Models

    def knn_model(self):
        # Train the classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=10)
        knn_classifier.fit(self.X_train, self.y_train)

        # make prediction
        prediction = knn_classifier.predict(self.X_test)
        predict_proba = knn_classifier.predict_proba(self.X_test)
        result = metrics(self.y_test, prediction, predict_proba[:, 1])

        return result

    def random_forest_model(self):
        # Train the classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=95)
        rf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = rf.predict(self.X_test)
        predict_proba = rf.predict_proba(self.X_test)

        result = metrics(self.y_test, predictions, predict_proba[:, 1])

        return result

    def svm_model(self):
        # Train the classifier
        svm_clf = svm.SVC(kernel='linear', C=1.0, probability=True)
        svm_clf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = svm_clf.predict(self.X_test)
        predict_proba = svm_clf.predict_proba(self.X_test)

        result = metrics(self.y_test, predictions, predict_proba[:, 1])

        return result

    def light_gbm(self):
        parameters = {
            "objective": "binary",  # binary_logloss/multiclass
            # droptrees#gbdt(decision trees)/goss(one side sampling(DO NOT Understand))
            "boosting_type": "dart",
            "is_unbalance": True,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,  # sub samples or bagging to reduce overfitting for each model
            "metric": 'binary_error',
            "num_leaves": 7,  # NEEDS RESEARCH
            # "early_stopping_rounds":10 :NEEDS RESEARCH
            "force_row_wise": True  # NEEDS RESEARCH
        }
        # Initialising Light GBM
        lgb_clf = lgbm.LGBMRegressor(**parameters)
        lgb_clf.fit(self.X_train, self.y_train)

        # Predicting
        predictions = lgb_clf.predict(
            self.X_test, num_iteration=lgb_clf.best_iteration_)
        binary_predictions = (predictions > 0.5).astype(int)
        # predict_proba = lgb_clf.predict_proba(self.X_test)

        # Evaluating Model
        results = metrics(self.y_test, binary_predictions, predictions)

        return results
