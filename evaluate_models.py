"""
MediScan AI - Model Evaluation and Visualization Script
Generates: Confusion Matrix, Training Curves, Classification Report
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST, INFO

# Create results directory
results_dir = "./evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# ============================================================================
# PART 1: SYMPTOM-BASED MODEL EVALUATION
# ============================================================================
print("=" * 60)
print("PART 1: SYMPTOM-BASED DISEASE PREDICTION MODEL")
print("=" * 60)

# Load data
data_path = "./datasets/DiseaseAndSymptoms.csv"
data = pd.read_csv(data_path)

# Preprocessing
cols = data.columns[1:]
for col in cols:
    data[col] = data[col].str.strip()

# Get all symptoms
raw_symptoms = data[cols].values.ravel('K')
all_symptoms = sorted(list(set([s for s in raw_symptoms if isinstance(s, str)])))

# Create binary matrix
X = pd.DataFrame(0, index=np.arange(len(data)), columns=all_symptoms)
for i in range(len(data)):
    row_symptoms = data.iloc[i, 1:].dropna().values
    X.loc[i, row_symptoms] = 1

# Encode labels
le = LabelEncoder()
y = le.fit_transform(data['Disease'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total Samples: {len(data)}")
print(f"Features (Symptoms): {len(all_symptoms)}")
print(f"Classes (Diseases): {len(le.classes_)}")
print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Random Forest Accuracy: {accuracy * 100:.2f}%")

# Save Classification Report
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(results_dir, "symptom_classification_report.csv"))
print(f"✓ Classification report saved")

# Confusion Matrix (Top 20 diseases)
cm = confusion_matrix(y_test, y_pred)
top_classes = np.argsort(np.bincount(y_test))[-20:]  # Top 20 most common

plt.figure(figsize=(16, 14))
cm_subset = cm[np.ix_(top_classes, top_classes)]
sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_[top_classes],
            yticklabels=le.classes_[top_classes])
plt.title('Confusion Matrix - Symptom Model (Top 20 Diseases)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "symptom_confusion_matrix.png"), dpi=150)
plt.close()
print(f"✓ Confusion matrix saved")

# Feature Importance
feature_importance = pd.DataFrame({
    'symptom': all_symptoms,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(30)

plt.figure(figsize=(12, 10))
plt.barh(feature_importance['symptom'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Top 30 Most Important Symptoms', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "symptom_feature_importance.png"), dpi=150)
plt.close()
print(f"✓ Feature importance chart saved")

# ============================================================================
# PART 2: VISION MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("PART 2: VISION-BASED SKIN DISEASE PREDICTION MODEL")
print("=" * 60)

# Load data
info = INFO['dermamnist']
n_classes = len(info['label'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

print("Loading DermaMNIST dataset...")
train_dataset = DermaMNIST(split='train', transform=transform, download=True)
test_dataset = DermaMNIST(split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train Samples: {len(train_dataset)}")
print(f"Test Samples: {len(test_dataset)}")
print(f"Classes: {n_classes}")

# Define Model
class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training with history
epochs = 10
train_losses = []
train_accuracies = []
test_accuracies = []

print("\nTraining Vision Model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).long().squeeze()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).long().squeeze()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

print(f"\n✓ Final Test Accuracy: {test_accuracies[-1]:.2f}%")

# Plot Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss Curve
axes[0].plot(range(1, epochs+1), train_losses, 'b-o', linewidth=2, markersize=6)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Curve', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Accuracy Curves
axes[1].plot(range(1, epochs+1), train_accuracies, 'b-o', label='Train Accuracy', linewidth=2)
axes[1].plot(range(1, epochs+1), test_accuracies, 'r-s', label='Test Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Train vs Test Accuracy', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "vision_training_curves.png"), dpi=150)
plt.close()
print(f"✓ Training curves saved")

# Vision Confusion Matrix
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy().squeeze())

label_names = [info['label'][str(i)] for i in range(n_classes)]
cm_vision = confusion_matrix(all_targets, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_vision, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.title('Confusion Matrix - Vision Model (Skin Diseases)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "vision_confusion_matrix.png"), dpi=150)
plt.close()
print(f"✓ Vision confusion matrix saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)
print(f"\nResults saved to: {results_dir}/")
print(f"  - symptom_classification_report.csv")
print(f"  - symptom_confusion_matrix.png")
print(f"  - symptom_feature_importance.png")
print(f"  - vision_training_curves.png")
print(f"  - vision_confusion_matrix.png")

print(f"\n📊 SYMPTOM MODEL: {accuracy * 100:.2f}% accuracy on {len(le.classes_)} diseases")
print(f"📷 VISION MODEL: {test_accuracies[-1]:.2f}% accuracy on {n_classes} skin conditions")
