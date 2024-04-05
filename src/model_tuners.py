import numpy as np
import matplotlib.pyplot as plt
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
        explanation = self.explainer.explain_instance(sample, pred_prob,
                                             num_features=num_features,
                                             top_labels=top_labels)
        return explanation

    def plot_feature_importance(self, model_result, file_name, top_n=10, file_format='png'):
        if isinstance(model_result, tuple):
            feature_importance = model_result[2]
            if feature_importance is not None:
                top_indices = np.argsort(feature_importance)[::-1][:top_n]
                top_features = [self.feature_names[i] for i in top_indices]
                top_importances = feature_importance[top_indices]

                plt.figure(figsize=(10, 6))
                plt.bar(range(top_n), top_importances, align='center')
                plt.xticks(range(top_n), top_features, rotation=45, ha='right')
                plt.xlabel('Feature')
                plt.ylabel('Importance')
                plt.title('Top {} Feature Importances'.format(top_n))
                file_path = f"{file_name}.{file_format}"
                plt.savefig(file_path)
                plt.close()
            else:
                print("Feature importances not available for this model.")
        else:
            print("Invalid model result format.")

    def calculate_feature_importance(self, model_result):
        if isinstance(model_result, tuple):
            feature_importance = model_result[2]
            if feature_importance is not None:
                feature_importance_list = [(feature, importance) for feature, importance in zip(self.feature_names, feature_importance)]
                feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
                return feature_importance_list
            else:
                print("Feature importances not available for this model.")
                return {}
        else:
            print("Invalid model result format.")
            return {}
