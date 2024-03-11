from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score


class TrainModel():

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def svm_model(self):
        clf = svm.SVC(kernel='linear', C=1.0)
        # Train the classifier
        clf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions)

        return accuracy, f1, precision

    def logistic_regression_model(self):
        clf = LogisticRegression(random_state=0)
        # Train the classifier
        clf.fit(self.X_train, self.y_train)

        # make prediction
        prediction = clf.predict(self.X_test)

        # evaluate accuracy
        accuracy = accuracy_score(self.y_test, prediction)
        precision = precision_score(self.y_test, prediction)

        return accuracy, precision

    def random_forest_model(self):
        # Create a Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the classifier
        clf.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = clf.predict(self.X_test)

        # Evaluate precision
        precision = precision_score(self.y_test, predictions)
        accuracy = accuracy_score(self.y_test, predictions)
        f_score = f1_score(self.y_test, predictions, average="binary")

        return precision, accuracy, f_score
