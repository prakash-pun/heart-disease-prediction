from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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
        
        #Metrics
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        f_score= f1_score(self.y_test, predictions)

        return accuracy, precision, recall, f_score

    def logistic_regression_model(self):
        # Train the classifier
        logreg = LogisticRegression(random_state=0)
        logreg.fit(self.X_train, self.y_train)

        # make prediction
        prediction = logreg.predict(self.X_test)

        # evaluate metrics
        accuracy = accuracy_score(self.y_test, prediction)
        precision = precision_score(self.y_test, prediction)
        recall = recall_score(self.y_test, prediction)
        f_score= f1_score(self.y_test, prediction)
        
        return accuracy, precision, recall, f_score

    def random_forest_model(self):
        # Train the classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=95)
        rf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = rf.predict(self.X_test)

        # Evaluate metrics
        precision = precision_score(self.y_test, predictions)
        accuracy = accuracy_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        f_score= f1_score(self.y_test, predictions)

        return accuracy, precision, recall, f_score
    
    def xg_boost(self):       
        # Define parameters for XGBoost
        params = {
            'booster': 'gbtree',
            'learning_rate': 0.15,
            'n_estimators':200,
            'subsample': 0.8,
            'max_depth': 3, #Tree Depth
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'merror'  # Evaluation metric
        }
                
        #XGB CLF
        xgb_clf=xgb.XGBClassifier(**params)
        xgb_clf.fit(self.X_train,self.y_train)
        
        # Make predictions
        predictions_clf = xgb_clf.predict(self.X_test)
        binary_predictions_clf = (predictions_clf > 0.5).astype(int)

        # Calculate metrics
        accuracy_clf=accuracy_score(self.y_test,binary_predictions_clf)
        precision = precision_score(self.y_test, binary_predictions_clf)
        recall = recall_score(self.y_test, binary_predictions_clf)
        f_score= f1_score(self.y_test, binary_predictions_clf)
        
        return accuracy_clf, precision, recall, f_score