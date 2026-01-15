"""
MediScan AI - Enhanced Vision Model Training
Features: Deeper CNN, BatchNorm, Dropout, Data Augmentation, 
          Learning Rate Scheduling, Temperature Scaling
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST, INFO
import joblib
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
train_batch_size = 64
test_batch_size = 64
lr = 0.001
epochs = 30  # Increased from 5
patience = 5  # Early stopping patience
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

print("=" * 60)
print("MediScan AI - Enhanced Vision Model Training")
print("=" * 60)

# ============================================================================
# DATA LOADING WITH AUGMENTATION
# ============================================================================
info = INFO['dermamnist']
n_classes = len(info['label'])

# Enhanced data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# Standard transform for testing (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

print("\n📥 Loading DermaMNIST dataset...")
train_dataset = DermaMNIST(split='train', transform=train_transform, download=True)
val_dataset = DermaMNIST(split='val', transform=test_transform, download=True)
test_dataset = DermaMNIST(split='test', transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

print(f"   Train Samples: {len(train_dataset)}")
print(f"   Validation Samples: {len(val_dataset)}")
print(f"   Test Samples: {len(test_dataset)}")
print(f"   Classes: {n_classes}")

# ============================================================================
# ENHANCED CNN MODEL
# ============================================================================
class EnhancedCNN(nn.Module):
    """
    Deeper CNN with BatchNorm and Dropout for better feature extraction
    and regularization to improve model confidence.
    """
    def __init__(self, n_classes):
        super(EnhancedCNN, self).__init__()
        
        # Block 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Block 3: 128 -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Block 4: 256 -> 512 channels
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout2d(0.25)
        
        # Classifier - after 4 pooling ops: 28x28 -> 14 -> 7 -> 3 -> 1
        # For DermaMNIST (28x28): After 3 pools -> 3x3, after 4 -> 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Adaptive pooling and classifier
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ============================================================================
# TEMPERATURE SCALING FOR CALIBRATION
# ============================================================================
class TemperatureScaledModel(nn.Module):
    """
    Wraps the trained model with temperature scaling for better
    probability calibration and more reliable confidence scores.
    """
    def __init__(self, model, temperature=1.5):
        super(TemperatureScaledModel, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature
    
    def calibrate(self, val_loader, device, max_iter=50, lr=0.01):
        """Optimize temperature on validation set"""
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        # Collect all validation logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels.squeeze())
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device).long()
        
        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_temp():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_temp)
        
        return self.temperature.item()


# ============================================================================
# TRAINING SETUP
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")

model = EnhancedCNN(n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"   Model Parameters: {total_params:,}")

# ============================================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================================
print(f"\n🏋️  Training for {epochs} epochs with early stopping (patience={patience})...")
print("-" * 60)

best_val_acc = 0.0
patience_counter = 0
train_losses = []
val_accuracies = []

for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).long().squeeze()
        
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
    
    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long().squeeze()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), os.path.join(models_dir, "best_vision_model.pth"))
    else:
        patience_counter += 1
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"   Epoch [{epoch+1:2d}/{epochs}] Loss: {train_loss:.4f} | "
          f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f} | Best: {best_val_acc:.2f}%")
    
    if patience_counter >= patience:
        print(f"\n   ⚠️ Early stopping triggered at epoch {epoch + 1}")
        break

# Load best model
model.load_state_dict(torch.load(os.path.join(models_dir, "best_vision_model.pth")))

# ============================================================================
# TEMPERATURE SCALING CALIBRATION
# ============================================================================
print("\n" + "-" * 60)
print("🎯 Applying Temperature Scaling for Calibration...")
print("-" * 60)

temp_model = TemperatureScaledModel(model).to(device)
optimal_temp = temp_model.calibrate(val_loader, device)
print(f"   Optimal Temperature: {optimal_temp:.4f}")

# ============================================================================
# FINAL EVALUATION ON TEST SET
# ============================================================================
print("\n" + "-" * 60)
print("📊 Final Evaluation on Test Set...")
print("-" * 60)

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

test_accuracy = 100 * correct / total
all_probs = np.concatenate(all_probs, axis=0)
avg_confidence = np.max(all_probs, axis=1).mean()

print(f"   ✓ Test Accuracy: {test_accuracy:.2f}%")
print(f"   ✓ Average Confidence: {avg_confidence * 100:.2f}%")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 60)
print("📦 Saving Models...")
print("=" * 60)

# Save temperature-scaled model state
torch.save({
    'model_state_dict': model.state_dict(),
    'temperature': temp_model.temperature.item(),
    'n_classes': n_classes
}, os.path.join(models_dir, "vision_disease_model.pth"))

# Save label map
label_map = {int(k): v for k, v in info['label'].items()}
joblib.dump(label_map, os.path.join(models_dir, "vision_label_map.joblib"))

print(f"   ✓ Enhanced vision model saved")
print(f"   ✓ Label map saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"📊 Test Accuracy: {test_accuracy:.2f}%")
print(f"🎯 Average Confidence: {avg_confidence * 100:.2f}%")
print(f"🌡️  Calibration Temperature: {optimal_temp:.4f}")
print(f"📁 Models saved to: {models_dir}/")
