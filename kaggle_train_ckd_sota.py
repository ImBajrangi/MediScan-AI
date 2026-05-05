import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ============================================================================
# 🔍 AUTOMATIC PATH DISCOVERY (Kaggle Specific)
# ============================================================================
def find_kaggle_paths():
    print("🔎 Searching for datasets in /kaggle/input...")
    
    # 1. Find the CSV file (Prioritize the larger dataset if available)
    csv_search = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    csv_path = None
    
    # Target specific large dataset if present
    big_dataset_candidates = [
        "Chronic_Kidney_Dsease_data.csv", 
        "Chronic_Kidney_Disease_data.csv",
        "kidney_disease.csv"
    ]
    
    for candidate in big_dataset_candidates:
        match = next((p for p in csv_search if candidate in p), None)
        if match:
            csv_path = match
            print(f"💎 High-capacity dataset detected: {csv_path}")
            break
            
    if not csv_path:
        csv_path = next((p for p in csv_search if "kidney" in p.lower()), None)
    
    # 2. Find the Vision Root (Searching for the folder containing 'Normal', 'Cyst', etc.)
    vision_root = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "Normal" in dirs and "Cyst" in dirs and "Stone" in dirs:
            vision_root = root
            break
            
    if not csv_path or not vision_root:
        # Fallback to hardcoded paths if auto-discovery fails
        print("⚠️ Auto-search failed or partial, checking common fallbacks...")
        fallbacks = [
            "/kaggle/input/kidney-disease-raw/kidney_disease_raw.csv",
            "/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        ]
        if os.path.exists(fallbacks[0]): csv_path = fallbacks[0]
        if os.path.exists(fallbacks[1]): vision_root = fallbacks[1]
    
    if not csv_path: raise FileNotFoundError("❌ Could not find kidney CSV dataset. Please add it to the notebook.")
    if not vision_root: raise FileNotFoundError("❌ Could not find CT Kidney image dataset. Please add it to the notebook.")
    
    print(f"✅ Found CSV: {csv_path}")
    print(f"✅ Found Images: {vision_root}")
    return csv_path, vision_root

# ============================================================================
# 🏗️ ARCHITECTURE (TabResNet + ResNet-50)
# ============================================================================
class TabResBlock(nn.Module):
    """Residual Block for Tabular Data Analysis"""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    def forward(self, x): 
        return self.relu(self.fc(x)) + x 

class MultiModalResNet(nn.Module):
    """Combined SOTA Pipeline: ResNet-50 (Vision) + TabularResNet (Clinical)"""
    def __init__(self, n_tab, n_clin, n_vis):
        super().__init__()
        print("🏗️ Initializing ResNet-50 Vision Backbone...")
        # NOTE: If internet is OFF in Kaggle Settings, this will fail.
        # Use 'Internet ON' in the right sidebar.
        try:
            self.vision_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception as e:
            print(f"⚠️ Warning: Could not download weights ({e}). Training from scratch.")
            self.vision_model = models.resnet50(weights=None)
            
        vis_features = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Sequential(
            nn.Linear(vis_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Robust MLP for Tabular Biomarkers
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
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        self.clinical_head = nn.Linear(256, n_clin)
        self.vision_head = nn.Linear(256, n_vis)

    def forward(self, tab, img):
        # Force float conversion to ensure CUDA compatibility across all kernels
        tab_feat = self.tab_model(tab.float())
        vis_feat = self.vision_model(img)
        combined = torch.cat([tab_feat, vis_feat], dim=1)
        fused = self.fusion(combined)
        return self.clinical_head(fused), self.vision_head(fused)

# ============================================================================
# 📊 DATA & TRAINING
# ============================================================================
class CKDDataset(Dataset):
    def __init__(self, tab_data, img_paths, clin_labels, vis_labels, tr):
        self.tab_data = torch.FloatTensor(tab_data)
        self.img_paths, self.clin_labels, self.vis_labels, self.tr = img_paths, clin_labels, vis_labels, tr
    def __len__(self): return len(self.tab_data)
    def __getitem__(self, idx):
        t = self.tab_data[idx]
        i_idx = idx % len(self.img_paths)
        img = Image.open(self.img_paths[i_idx]).convert('RGB')
        return {'tabular': t, 'image': self.tr(img), 'clinical_label': self.clin_labels[idx], 'vision_label': self.vis_labels[i_idx]}

def run_kaggle_training():
    csv_path, vision_root = find_kaggle_paths()
    models_dir = "/kaggle/working/models"
    os.makedirs(models_dir, exist_ok=True)

    def load_and_prep(DATA_PATH):
        print("🔄 Preprocessing Data...")
        df = pd.read_csv(DATA_PATH)
        
        # Drop ID columns
        id_cols = ['PatientID', 'id', 'Unnamed: 0']
        for col in id_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
                print(f"🗑️ Dropped ID column: {col}")

        # Identify Target
        target_candidates = ['Diagnosis', 'class', 'target']
        target_col = next((c for c in target_candidates if c in df.columns), df.columns[-1])
        print(f"🎯 Using Target Column: {target_col}")

        # Scale & Encode
        for col in df.columns:
            if col == target_col: continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str).str.lower().str.strip())
        
        X_tab = StandardScaler().fit_transform(df.drop([target_col], axis=1))
        c_labels = LabelEncoder().fit_transform(df[target_col].astype(str).str.lower().str.strip())
        
        print(f"✅ Features processed: {X_tab.shape[1]}")
        print(f"✅ Target categories: {len(np.unique(c_labels))}")
        return X_tab, c_labels

    X_tab, c_labels = load_and_prep(csv_path)
    
    img_paths, v_labels = [], []
    classes = [c for c in sorted(os.listdir(vision_root)) if os.path.isdir(os.path.join(vision_root, c))]
    c_to_i = {n: i for i, n in enumerate(classes)}
    for c in classes:
        p = os.path.join(vision_root, c)
        for img in os.listdir(p):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(p, img))
                v_labels.append(c_to_i[c])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    Xtr, Xte, ytr, yte = train_test_split(X_tab, c_labels, test_size=0.2, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # drop_last=True is critical for BatchNorm stability on some CUDA versions
    train_loader = DataLoader(
        CKDDataset(Xtr, img_paths, ytr, v_labels, transform), 
        batch_size=32, 
        shuffle=True,
        drop_last=True
    )
    
    model = MultiModalResNet(X_tab.shape[1], len(np.unique(c_labels)), len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 50 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Starting training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            tab, img = batch['tabular'].to(device), batch['image'].to(device)
            yc, yv = batch['clinical_label'].to(device), batch['vision_label'].to(device)
            
            optimizer.zero_grad()
            out_c, out_v = model(tab, img)
            loss = criterion(out_c, yc) + 0.5 * criterion(out_v, yv)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        print(f"✅ Epoch [{epoch+1}/{epochs}] - Avg Loss: {total_loss/len(train_loader):.4f}")

    save_path = os.path.join(models_dir, "ckd_sota_resnet.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✨ SUCCESS: SOTA weights saved to {save_path}")

if __name__ == "__main__":
    run_kaggle_training()
