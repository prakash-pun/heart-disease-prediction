import lime
import lime.lime_tabular


def train_lime_explainer(X_train, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                       feature_names=feature_names,
                                                       class_names=['0', '1'])
    return explainer


def explain_prediction(explainer, sample, pred_prob, num_features, top_labels=1):
    print("asdf", pred_prob)
    # pred_prob = model.predict_proba(sample.reshape(1, -1))
    explanation = explainer.explain_instance(sample, pred_prob,
                                             num_features=num_features,
                                             top_labels=top_labels)
    return explanation
