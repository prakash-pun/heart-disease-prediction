import os
import joblib
import numpy as np
from .train_model import metrics


def get_dump_file(filename=""):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    path = os.path.join(project_dir, f'src/dump_model/{filename}')

    return path


def get_dump_model(filename):
    filepath = get_dump_file(filename)
    model = joblib.load(filepath)

    return model


class DumpTrainModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def logistic_regression_model(self):
        logreg = get_dump_model("logistic_model.joblib")

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

        return {"train": result_train, "test": result_test, "predict": logreg, "feature_importance": feature_importance, "feature_names": feature_names}

    def xg_boost(self):
        xgb_clf = get_dump_model("xg_boost_model.joblib")

        # Make predictions
        train_predict = xgb_clf.predict(self.X_train)
        train_predictions_clf = (train_predict > 0.5).astype(int)
        train_proba = xgb_clf.predict_proba(self.X_train)

        predictions_clf = xgb_clf.predict(self.X_test)
        binary_predictions_clf = (predictions_clf > 0.5).astype(int)
        predict_proba = xgb_clf.predict_proba(self.X_test)

        # Calculate metrics
        result_train = metrics(
            self.y_train, train_predictions_clf, train_proba[:, 1])
        result = metrics(self.y_test, binary_predictions_clf,
                         predict_proba[:, 1])

        # Calculate feature importances
        feature_importance = xgb_clf.feature_importances_
        feature_names = self.X_train.columns.tolist()

        return {"train": result_train, "test": result, "predict": xgb_clf, "feature_importance": feature_importance, "feature_names": feature_names}

    def gbm_model(self):
        gradient_boosting = get_dump_model("gbm_model.joblib")

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

        return {"train": result_train, "test": result, "feature_importance": feature_importance, "feature_names": feature_names}
