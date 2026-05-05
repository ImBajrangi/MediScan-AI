import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ============================================================================
# 🔍 ASSET DISCOVERY
# ============================================================================
def find_assets():
    print("🔎 Locating Kaggle Assets...")
    csv_search = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    csv_path = next((p for p in csv_search if "kidney" in p.lower()), None)
    
    vision_root = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "Normal" in dirs and "Cyst" in dirs and "Stone" in dirs:
            vision_root = root
            break
            
    model_path = "/kaggle/working/models/ckd_sota_resnet.pth"
    return csv_path, vision_root, model_path

# ============================================================================
# 🏗️ ARCHITECTURE (TabResNet + ResNet-50)
# ============================================================================
class TabResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    def forward(self, x): 
        return self.relu(self.fc(x)) + x 

class MultiModalResNet(nn.Module):
    def __init__(self, n_tab, n_clin, n_vis):
        super().__init__()
        # Load backbone without weights for evaluation
        self.vision_model = models.resnet50(weights=None)
        vis_features = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Sequential(
            nn.Linear(vis_features, 256),
            nn.ReLU()
        )
        
        self.tab_model = nn.Sequential(
            nn.Linear(n_tab, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            TabResBlock(128),
            nn.Linear(128, 128)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        self.clinical_head = nn.Linear(256, n_clin)
        self.vision_head = nn.Linear(256, n_vis)

    def forward(self, tab, img):
        tab_feat = self.tab_model(tab.float())
        vis_feat = self.vision_model(img)
        fused = self.fusion(torch.cat([tab_feat, vis_feat], dim=1))
        return self.clinical_head(fused), self.vision_head(fused)

# ============================================================================
# 📊 DATA LOADER
# ============================================================================
class CKDDataset(Dataset):
    def __init__(self, tab_data, img_paths, clin_labels, vis_labels, tr):
        self.tab_data = torch.FloatTensor(tab_data)
        self.img_paths, self.clin_labels, self.vis_labels, self.tr = img_paths, clin_labels, vis_labels, tr
    def __len__(self): return len(self.tab_data)
    def __getitem__(self, idx):
        t = self.tab_data[idx]
        img = Image.open(self.img_paths[idx % len(self.img_paths)]).convert('RGB')
        return {'tabular': t, 'image': self.tr(img), 'clinical_label': self.clin_labels[idx]}

# ============================================================================
# 📈 MEGA VISUALIZER
# ============================================================================
def run_mega_visualizer():
    csv_path, vision_root, model_path = find_assets()
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}")
        return

    # 1. PREP TABULAR DATA
    print("📊 Loading Tabular Data...")
    df = pd.read_csv(csv_path)
    for c in ['PatientID', 'id', 'Unnamed: 0']:
        if c in df.columns: df.drop(c, axis=1, inplace=True)
        
    target_col = next((c for c in ['Diagnosis', 'class', 'target'] if c in df.columns), df.columns[-1])
    
    # Preprocessing
    le_clin = LabelEncoder()
    y_full = le_clin.fit_transform(df[target_col].astype(str).str.lower().str.strip())
    
    feature_names = [c for c in df.columns if c != target_col]
    for c in feature_names:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str).str.lower().str.strip())
            
    X_full = StandardScaler().fit_transform(df[feature_names])
    _, X_te, _, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    # 2. PREP VISION DATA
    print("🖼️ Loading Vision Data...")
    classes = sorted([c for c in os.listdir(vision_root) if os.path.isdir(os.path.join(vision_root, c))])
    img_paths = []
    for c in classes:
        p = os.path.join(vision_root, c)
        img_paths.extend([os.path.join(p, i) for i in os.listdir(p) if i.lower().endswith(('.png', '.jpg', '.jpeg'))])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_loader = DataLoader(CKDDataset(X_te, img_paths, y_te, None, transform), batch_size=32, shuffle=False)

    # 3. LOAD MODEL
    print("🚀 Initializing SOTA Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalResNet(X_full.shape[1], len(np.unique(y_full)), len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. PLOTS
    print("🎨 Generating Visualizations...")
    sns.set_theme(style="whitegrid")
    
    # --- Page 1: Dataset Distributions ---
    plt.figure(figsize=(18, 6))
    
    # Clinical Balance
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_full, palette="viridis")
    plt.title("Distribution of Clinical Stages", fontsize=14, fontweight='bold')
    plt.xlabel("Encoded Stage")

    # Vision Balance
    plt.subplot(1, 2, 2)
    counts = [len([i for i in img_paths if c in i]) for c in classes]
    plt.pie(counts, labels=classes, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    plt.title("CT Scan Class Balance", fontsize=14, fontweight='bold')
    plt.show()

    # --- Page 2: Feature Importance (Tabular) ---
    plt.figure(figsize=(14, 8))
    # Extract weights from first tabular layer
    weights = model.tab_model[0].weight.abs().mean(dim=0).cpu().detach().numpy()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': weights[:len(feature_names)]})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="magma")
    plt.title("Top 20 Critical Biomarkers (Model Weighted)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # --- Page 3: Model Performance ---
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            tab, img = batch['tabular'].to(device), batch['image'].to(device)
            out_c, _ = model(tab, img)
            all_preds.extend(torch.argmax(out_c, dim=1).cpu().numpy())
            all_true.extend(batch['clinical_label'].numpy())

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Clinical Staging: Prediction vs Truth", fontsize=15, fontweight='bold')
    plt.xlabel("Predicted Stage")
    plt.ylabel("Actual Stage")
    plt.show()

    print("\n📝 COMPREHENSIVE CLASSIFICATION REPORT:")
    print(classification_report(all_true, all_preds))

if __name__ == "__main__":
    run_mega_visualizer()
