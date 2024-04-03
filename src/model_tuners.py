from lime.lime_tabular import LimeTabularExplainer


class ModelTuning:
    def __init__(self, X_train, feature_names):
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = self.train_lime_explainer()

    def train_lime_explainer(self):

        explainer = LimeTabularExplainer(self.X_train.values,
                                         feature_names=self.feature_names,
                                         class_names=['0', '1'],
                                         discretize_continuous=True)
        return explainer

    def explain_prediction(self, sample, pred_prob, num_features, top_labels=1):
        print("asdf", pred_prob)
        # pred_prob = model.predict_proba(sample.reshape(1, -1))
        explanation = self.explainer.explain_instance(sample, pred_prob,
                                             num_features=num_features,
                                             top_labels=top_labels)
        return explanation
