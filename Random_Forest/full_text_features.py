import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the CSV file

data = pd.read_csv('/content/drive/MyDrive/Maram/Text/Features_Extraction_from_text/Features_full_Text/LIWC/LIWC_fulltxt_with_labels.csv')

# Separate features (X) and labels (y)
X_combined = data.drop(['PID','CAI State'], axis=1).values  # Features without the label column
y_combined = data['CAI State'].values  # Labels column


# Initialize Random Forest model
rf_model = RandomForestClassifier()

# Set up Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Store metrics per fold
accuracies = []
precisions = []
recalls = []
f1s = []
confusion_matrices = []
predictions = []
true_labels = []

for train_index, test_index in kf.split(X_combined, y_combined):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = y_combined[train_index], y_combined[test_index]

     # Initialize StandardScaler
    scaler = StandardScaler()
    # Scale the features 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Fit the model on the training data
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)

    # Store individual fold metrics
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1s.append(f1_score(y_test, y_pred, average='weighted'))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # For cumulative confusion
    predictions.extend(y_pred)
    true_labels.extend(y_test)

print("Evaluation Results :")
print(f"Accuracy:  {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
print(f"Recall:    {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
print(f"F1 Score:  {np.mean(f1s):.2f} ± {np.std(f1s):.2f}")

# confusion matrices
total_confusion = sum(confusion_matrices)
print("\n Confusion Matrix:")
print(total_confusion)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6, 5))
sns.heatmap(total_confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.title("Cumulative Confusion Matrix (Heatmap)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

