from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def Lsvc(X_train, X_test, y_train, y_test):

    svc = LinearSVC(max_iter=1000)
    svc.fit(X_train, y_train)
    y_pred1 = svc.predict(X_test)
    y_prob1 = svc._predict_proba_lr(X_test)
    return y_pred1,y_prob1


def train_naive_bayes(X_train, X_test, y_train):

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred2 = gnb.predict(X_test)
    y_prob2 = gnb.predict_proba(X_test)
    return y_pred2, y_prob2


def calc_metric(y_true, y_pred, y_prob=None):
    # created a dictionary
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    metrics['precision'] = precision_score(y_true, y_pred)

    metrics['recall'] = recall_score(y_true, y_pred)

    metrics['f1_score'] = f1_score(y_true, y_pred)

    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])

    return metrics