import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os

# 1. Simple CNN Architecture for Kidney Imaging
class KidneyCNN(nn.Module):
    def __init__(self, n_classes=3):
        super(KidneyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Synthetic Dataset for Training
class SyntheticKidneyDataset(Dataset):
    def __init__(self, length=300):
        self.length = length
        self.data = torch.randn(length, 1, 28, 28)
        self.labels = torch.randint(0, 3, (length,))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 3. Training Script
def train_vision_model():
    print("Initializing synthetic CKD vision model training...")
    dataset = SyntheticKidneyDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = KidneyCNN(n_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "ckd_vision_model.pth"))
    print(f"Model saved to {models_dir}/ckd_vision_model.pth")

if __name__ == "__main__":
    train_vision_model()
