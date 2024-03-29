import lime
from lime.lime_tabular import LimeTabularExplainer

def train_lime_explainer(X_train, feature_names):
    explainer = LimeTabularExplainer(X_train.values,
                                     feature_names=feature_names,
                                     class_names=['0', '1'],
                                     discretize_continuous=True)
    return explainer

def explain_prediction(explainer, sample, model, num_features, top_labels = 1):
    pred_prob = model.predict_proba(sample)
    explanation = explainer.explain_instance(sample, pred_prob,
                                             num_features=num_features,
                                             top_labels=top_labels)
    return explanation
