import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the CSV file
data = pd.read_csv('/content/drive/MyDrive/Maram/Text/Features_Extraction_from_text/Features_chunk_Text/LIWC/LIWC-22-cleaned_Chtxt_with_labels.csv')
# Separate features (X) and labels (y)
X_combined = data.drop(['filename', 'CAI State'], axis=1).values
y_combined = data['CAI State'].values  # Labels column

# Define person_ids
person_ids = data['filename']


# Define MLP class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y):
        m = y.shape[0]
        error = self.a2 - y
        d_output = error * self.sigmoid_derivative(self.a2)
        error_hidden = np.dot(d_output, self.w2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        self.w2 -= self.learning_rate * np.dot(self.a1.T, d_output) / m
        self.b2 -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True) / m
        self.w1 -= self.learning_rate * np.dot(x.T, d_hidden) / m
        self.b1 -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True) / m

    def train(self, x, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y)

    def predict(self, x):
        output = self.forward(x)
        return (output > 0.5).astype(int)

# Cross-validation
gkf = GroupKFold(n_splits=10)
accuracies, precisions, recalls, f1s, confusion_matrices = [], [], [], [], []

for train_idx, test_idx in gkf.split(X_combined, y_combined, groups=person_ids):
    X_train, X_test = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = y_combined[train_idx], y_combined[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape labels for MLP
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Train MLP
    mlp = MLP(input_size=X_train.shape[1], hidden_size=10, output_size=1, learning_rate=0.1)
    mlp.train(X_train, y_train, epochs=1000)

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

