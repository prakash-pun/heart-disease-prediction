from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def KNN_model(y_true, y_pred):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision
    precision = precision_score(y_true, y_pred, average='weighted')
    
    # Recall
    recall = recall_score(y_true, y_pred, average='weighted')
    
    # F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, cm

# Example usage:
# Assuming y_true and y_pred are your true and predicted labels respectively
y_true = [0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 1, 0, 1]

accuracy, precision, recall, f1, cm = evaluate_classification(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)

return accuracy_score,precision_score,recall_score,f1_score,confusion_matrix