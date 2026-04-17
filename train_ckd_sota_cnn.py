import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

RESULTS_DIR = "./evaluation_results/ckd_sota_research"
MODELS_DIR = "./models"
DATA_PATH = "datasets/kidney_disease_raw.csv"
VISION_ROOT = "datasets/CKD/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif"
})
MEDICAL_PALETTE = ['#58a6ff', '#3fb950', '#f85149', '#d29922', '#bc8cff']

class CKDMultiModalDataset(Dataset):
    """
    Combined Dataset for Clinical (Tabular) and Vision (Image) Research.
    Uses late-fusion support.
    """
    def __init__(self, tabular_data, image_paths, clinical_labels, vision_labels, transform=None):
        self.tabular_data = torch.FloatTensor(tabular_data)
        self.image_paths = image_paths
        self.clinical_labels = torch.LongTensor(clinical_labels)
        self.vision_labels = torch.LongTensor(vision_labels)
        self.transform = transform

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        tab = self.tabular_data[idx]
        
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return {
            'tabular': tab,
            'image': img,
            'clinical_label': self.clinical_labels[idx],
            'vision_label': self.vision_labels[img_idx]
        }

def prepare_research_data():
    print("🔬 Preparing Multi-Modal Research Dataset...")
    
    df = pd.read_csv(DATA_PATH, index_col=0)
    
    for col in df.columns:
        if col == 'class': continue
        s_numeric = pd.to_numeric(df[col], errors='coerce')
        if s_numeric.isna().sum() < len(df) * 0.9:
            df[col] = s_numeric.fillna(s_numeric.median())
        else:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str).str.strip().str.lower())

    X_tab = df.drop(['class'], axis=1)
    clinical_labels = LabelEncoder().fit_transform(df['class'].astype(str).str.strip().str.lower())
    
    scaler = StandardScaler()
    X_tab_scaled = scaler.fit_transform(X_tab)
    
    image_paths = []
    vision_labels = []
    classes = sorted(os.listdir(VISION_ROOT))
    if '.DS_Store' in classes: classes.remove('.DS_Store')
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(VISION_ROOT, cls)
        if not os.path.isdir(cls_path): continue
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_path, img_name))
                vision_labels.append(class_to_idx[cls])

    X_train_tab, X_test_tab, y_train_clin, y_test_clin = train_test_split(
        X_tab_scaled, clinical_labels, test_size=0.2, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = CKDMultiModalDataset(X_train_tab, image_paths, y_train_clin, vision_labels, transform=transform)
    test_ds = CKDMultiModalDataset(X_test_tab, image_paths, y_test_clin, vision_labels, transform=transform)
    
    return train_ds, test_ds, X_tab.shape[1], len(classes), len(np.unique(clinical_labels))

class TabResBlock(nn.Module):
    """Residual Block for Tabular Data Analysis"""
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ln(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual

class TabularResNet(nn.Module):
    """Deep Residual MLP for tabular biomarkers"""
    def __init__(self, n_features, hidden_dims=[128, 256, 128], out_dim=128):
        super().__init__()
        layers = []
        curr_dim = n_features
        for h in hidden_dims:
            layers.append(TabResBlock(curr_dim, h))
            curr_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

class MultiModalResNet(nn.Module):
    """
    Combined SOTA (Non-Transformer) Pipeline.
    ResNet-50 (Vision) + TabularResNet (Clinical).
    """
    def __init__(self, n_tab_features, n_clin_classes, n_vis_classes):
        super().__init__()
        
        print("🏗️ Loading ResNet-50 Vision Backbone...")
        self.vision_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        vis_features = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Sequential(
            nn.Linear(vis_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.tab_model = TabularResNet(n_tab_features)
        
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        self.clinical_head = nn.Linear(256, n_clin_classes)
        self.vision_head = nn.Linear(256, n_vis_classes)

    def forward(self, tab, img):
        tab_feat = self.tab_model(tab)
        vis_feat = self.vision_model(img)
        
        combined = torch.cat([tab_feat, vis_feat], dim=1)
        fused = self.fusion(combined)
        
        return self.clinical_head(fused), self.vision_head(fused)

def run_training():
    train_ds, test_ds, n_tab, n_vis_cls, n_clin_cls = prepare_research_data()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    
    model = MultiModalResNet(n_tab, n_clin_cls, n_vis_cls).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )
    
    history = {'train_loss': [], 'clin_acc': [], 'vis_acc': []}
    
    print("\n⚡ Starting SOTA Multi-Modal Training (ResNet-50 + TabResNet)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_clin, correct_vis = 0, 0
        total = 0
        
        for batch in train_loader:
            tab, img = batch['tabular'].to(device), batch['image'].to(device)
            y_clin, y_vis = batch['clinical_label'].to(device), batch['vision_label'].to(device)
            
            optimizer.zero_grad()
            out_clin, out_vis = model(tab, img)
            
            loss_clin = criterion(out_clin, y_clin)
            loss_vis = criterion(out_vis, y_vis)
            loss = loss_clin + 0.5 * loss_vis
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            correct_clin += (torch.argmax(out_clin, 1) == y_clin).sum().item()
            correct_vis += (torch.argmax(out_vis, 1) == y_vis).sum().item()
            total += y_clin.size(0)
            
        avg_loss = epoch_loss / len(train_loader)
        clin_acc = correct_clin / total
        vis_acc = correct_vis / total
        
        history['train_loss'].append(avg_loss)
        history['clin_acc'].append(clin_acc)
        history['vis_acc'].append(vis_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | Clin Acc: {clin_acc:.2%} | Vis Acc: {vis_acc:.2%}")

    model.eval()
    all_clin_preds, all_clin_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            tab, img = batch['tabular'].to(device), batch['image'].to(device)
            out_clin, _ = model(tab, img)
            all_clin_preds.extend(torch.argmax(out_clin, dim=1).cpu().numpy())
            all_clin_true.extend(batch['clinical_label'].numpy())

    f1 = f1_score(all_clin_true, all_clin_preds, average='macro')
    prec = precision_score(all_clin_true, all_clin_preds, average='macro', zero_division=0)
    rec = recall_score(all_clin_true, all_clin_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_clin_true, all_clin_preds)

    print(f"\n📊 Final Research Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "ckd_sota_resnet.pth"))
    print(f"✅ Model weights saved to {MODELS_DIR}/ckd_sota_resnet.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], color=MEDICAL_PALETTE[0], label='Loss')
    plt.title("Training Convergence")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['clin_acc'], color=MEDICAL_PALETTE[1], label='Clinical Acc')
    plt.plot(history['vis_acc'], color=MEDICAL_PALETTE[3], label='Vision Acc')
    plt.title("Staging Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "training_metrics.png"))
    plt.close()

    cm = confusion_matrix(all_clin_true, all_clin_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("CKD Clinical Staging Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    run_training()
