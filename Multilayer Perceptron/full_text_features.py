import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load CSV file
data = pd.read_csv('/content/drive/MyDrive/Maram/Text/Features_Extraction_from_text/Features_full_Text/LIWC/LIWC_fulltxt_with_labels.csv')

# Separate features (X) and labels (y)
X_combined = data.drop(['PID','CAI State'], axis=1).values  # Features without the label column
y_combined = data['CAI State'].values  # Labels column


# Cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies, precisions, recalls, f1s, confusion_matrices = [], [], [], [], []

for train_idx, test_idx in kf.split(X_combined, y_combined):
    X_train, X_test = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = y_combined[train_idx], y_combined[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ONE-TIME seeding
    np.random.seed(42)


    # Create and train the MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, activation='relu', learning_rate_init=0.01, random_state=42)

    # Train the model using the training data
    mlp.fit(X_train,y_train)

    # Predict and evaluate
    y_pred = mlp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# Print evaluation results
print("Evaluation Results:")
print(f"Accuracy:  {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
print(f"Recall:    {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
print(f"F1 Score:  {np.mean(f1s):.2f} ± {np.std(f1s):.2f}")



# confusion matrices across folds
total_confusion = sum(confusion_matrices)
print("\nCumulative Confusion Matrix:")
print(total_confusion)

plt.figure(figsize=(6, 5))
sns.heatmap(total_confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.title("Cumulative Confusion Matrix (Heatmap)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

