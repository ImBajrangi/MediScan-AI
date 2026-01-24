
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST, INFO

results_dir = "./evaluation_results"
os.makedirs(results_dir, exist_ok=True)

print("=" * 60)
print("PART 1: ENHANCED SYMPTOM-BASED DISEASE PREDICTION MODEL")
print("=" * 60)

data_path = "./datasets/DiseaseAndSymptoms.csv"
data = pd.read_csv(data_path)

cols = data.columns[1:]
for col in cols:
    data[col] = data[col].str.strip()

raw_symptoms = data[cols].values.ravel('K')
all_symptoms = sorted(list(set([s for s in raw_symptoms if isinstance(s, str)])))\

X = pd.DataFrame(0, index=np.arange(len(data)), columns=all_symptoms)
for i in range(len(data)):
    row_symptoms = data.iloc[i, 1:].dropna().values
    X.loc[i, row_symptoms] = 1

le = LabelEncoder()
y = le.fit_transform(data['Disease'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total Samples: {len(data)}")
print(f"Features (Symptoms): {len(all_symptoms)}")
print(f"Classes (Diseases): {len(le.classes_)}")
print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")

print("\n🔧 Training Enhanced Ensemble Model...")

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=22,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

xgb_model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=14,
    learning_rate=0.1,
    subsample=0.85,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_model)],
    voting='soft',
    weights=[1.2, 1.0]
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
avg_confidence = np.max(y_proba, axis=1).mean()
print(f"\n✓ Ensemble Accuracy: {accuracy * 100:.2f}%")
print(f"✓ Average Confidence: {avg_confidence * 100:.2f}%")

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(results_dir, "symptom_classification_report.csv"))
print(f"✓ Classification report saved")

cm = confusion_matrix(y_test, y_pred)
top_classes = np.argsort(np.bincount(y_test))[-20:]

plt.figure(figsize=(16, 14))
cm_subset = cm[np.ix_(top_classes, top_classes)]
sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_[top_classes],
            yticklabels=le.classes_[top_classes])
plt.title('Confusion Matrix - Enhanced Ensemble Model (Top 20 Diseases)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "symptom_confusion_matrix.png"), dpi=150)
plt.close()
print(f"✓ Confusion matrix saved")

rf.fit(X_train, y_train)
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

plt.figure(figsize=(10, 6))
max_probs = np.max(y_proba, axis=1)
plt.hist(max_probs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=avg_confidence, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_confidence:.3f}')
plt.xlabel('Prediction Confidence', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Symptom Model - Confidence Distribution', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "symptom_confidence_distribution.png"), dpi=150)
plt.close()
print(f"✓ Confidence distribution saved")

print("\n" + "=" * 60)
print("PART 2: ENHANCED VISION-BASED SKIN DISEASE PREDICTION MODEL")
print("=" * 60)

info = INFO['dermamnist']
n_classes = len(info['label'])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

print("Loading DermaMNIST dataset...")
train_dataset = DermaMNIST(split='train', transform=train_transform, download=True)
val_dataset = DermaMNIST(split='val', transform=test_transform, download=True)
test_dataset = DermaMNIST(split='test', transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train Samples: {len(train_dataset)}")
print(f"Validation Samples: {len(val_dataset)}")
print(f"Test Samples: {len(test_dataset)}")
print(f"Classes: {n_classes}")

class EnhancedCNN(nn.Module):
    def __init__(self, n_classes):
        super(EnhancedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout2d(0.25)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.dropout_conv(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout_conv(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout_conv(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class TemperatureScaledModel(nn.Module):
    def __init__(self, model, temperature=1.5):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN(n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

epochs = 25
train_losses = []
train_accuracies = []
val_accuracies = []
best_val_acc = 0

print("\n🏋️ Training Enhanced Vision Model...")
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
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).long().squeeze()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f}, Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")

temp_model = TemperatureScaledModel(model).to(device)

temp_model.eval()
correct = 0
total = 0
all_probs = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).long().squeeze()
        outputs = temp_model(inputs)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

test_acc = 100 * correct / total
all_probs = np.concatenate(all_probs, axis=0)
vision_avg_confidence = np.max(all_probs, axis=1).mean()

print(f"\n✓ Final Test Accuracy: {test_acc:.2f}%")
print(f"✓ Average Confidence: {vision_avg_confidence * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, epochs+1), train_losses, 'b-o', linewidth=2, markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Curve', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, epochs+1), train_accuracies, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
axes[1].plot(range(1, epochs+1), val_accuracies, 'r-s', label='Val Accuracy', linewidth=2, markersize=4)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Train vs Validation Accuracy', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "vision_training_curves.png"), dpi=150)
plt.close()
print(f"✓ Training curves saved")

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
plt.title('Confusion Matrix - Enhanced Vision Model (Skin Diseases)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "vision_confusion_matrix.png"), dpi=150)
plt.close()
print(f"✓ Vision confusion matrix saved")

plt.figure(figsize=(10, 6))
vision_max_probs = np.max(all_probs, axis=1)
plt.hist(vision_max_probs, bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(x=vision_avg_confidence, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {vision_avg_confidence:.3f}')
plt.xlabel('Prediction Confidence', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Vision Model - Confidence Distribution', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "vision_confidence_distribution.png"), dpi=150)
plt.close()
print(f"✓ Vision confidence distribution saved")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)
print(f"\nResults saved to: {results_dir}/")
print(f"  - symptom_classification_report.csv")
print(f"  - symptom_confusion_matrix.png")
print(f"  - symptom_feature_importance.png")
print(f"  - symptom_confidence_distribution.png")
print(f"  - vision_training_curves.png")
print(f"  - vision_confusion_matrix.png")
print(f"  - vision_confidence_distribution.png")

print(f"\n📊 SYMPTOM MODEL:")
print(f"   Accuracy: {accuracy * 100:.2f}%")
print(f"   Average Confidence: {avg_confidence * 100:.2f}%")
print(f"\n📷 VISION MODEL:")
print(f"   Accuracy: {test_acc:.2f}%")
print(f"   Average Confidence: {vision_avg_confidence * 100:.2f}%")
