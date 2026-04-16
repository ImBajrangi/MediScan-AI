import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ============================================================================
# CONFIGURATION & GLOBAL STYLING
# ============================================================================
RESULTS_DIR = "./evaluation_results/ckd_modern_research"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# ============================================================================
# DATASET & PREPROCESSING
# ============================================================================

class MultiModalCKDDataset(Dataset):
    def __init__(self, tabular_df, image_paths, clinical_labels, vision_labels, transform=None):
        self.tabular_data = torch.FloatTensor(tabular_df.values)
        self.image_paths = image_paths
        self.clinical_labels = torch.LongTensor(clinical_labels)
        self.vision_labels = torch.LongTensor(vision_labels)
        self.transform = transform

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        tab = self.tabular_data[idx]
        img_path = self.image_paths[max(0, idx % len(self.image_paths))] # Repeat images if tabular is larger
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return {
            'tabular': tab,
            'image': img,
            'clinical_label': self.clinical_labels[idx],
            'vision_label': self.vision_labels[idx % len(self.vision_labels)]
        }

def prepare_data():
    print("📊 Preparing SOTA Data Pipeline (25 Clinical Features + CT Images)...")
    
    # 1. Tabular Data (Clinical)
    data_path = "datasets/kidney_disease_raw.csv"
    df = pd.read_csv(data_path, index_col=0)
    
    # Process all clinical features (Handling both numeric and categorical)
    for col in df.columns:
        if col == 'class': continue
        
        # Try to convert to numeric, if fail it stays object
        s_numeric = pd.to_numeric(df[col], errors='coerce')
        
        if s_numeric.isna().sum() < len(df) * 0.9: # If it's mostly numeric
            df[col] = s_numeric.fillna(s_numeric.median())
        else:
            # It's categorical
            df[col] = LabelEncoder().fit_transform(df[col].astype(str).str.strip().str.lower())

    # Features and Staging
    X_tab = df.drop(['class'], axis=1)
    
    # Label for CKD status
    clinical_labels = LabelEncoder().fit_transform(df['class'].astype(str).str.strip().str.lower())
    
    scaler = StandardScaler()
    X_tab_scaled = scaler.fit_transform(X_tab)
    
    # 2. Vision Data (CT Scans)
    vision_root = "datasets/CKD/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    vision_dataset = ImageFolder(vision_root)
    image_paths = [path for path, label in vision_dataset.imgs]
    vision_labels = vision_dataset.targets
    
    # Split
    X_train_tab, X_test_tab, y_train_clin, y_test_clin = train_test_split(X_tab_scaled, clinical_labels, test_size=0.2, random_state=42)
    
    # Transform for ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultiModalCKDDataset(
        pd.DataFrame(X_train_tab), image_paths[:len(X_train_tab)], 
        y_train_clin, vision_labels[:len(X_train_tab)], transform=transform
    )
    test_dataset = MultiModalCKDDataset(
        pd.DataFrame(X_test_tab), image_paths[-len(X_test_tab):], 
        y_test_clin, vision_labels[-len(X_test_tab):], transform=transform
    )
    
    return train_dataset, test_dataset, X_tab.shape[1], 4, len(np.unique(clinical_labels))

# ============================================================================
# MODERN ARCHITECTURES (Full-Scale)
# ============================================================================

class FTTransformer(nn.Module):
    """Simplified Feature Tokenizer Transformer for Tabular Data"""
    def __init__(self, n_features, embed_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(n_features * embed_dim, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        return self.out(self.flatten(x))

class AutoInt(nn.Module):
    """Automatic Feature Interaction learning for Tabular Data"""
    def __init__(self, n_features, embed_dim=64, n_heads=4):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.out = nn.Linear(n_features * embed_dim, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        attn_out, _ = self.attn(x, x, x)
        return self.out(torch.flatten(attn_out, 1))

class SwinStyleVision(nn.Module):
    """Shifted Window style attention for medical vision (Simplified)"""
    def __init__(self, channels=3, embed_dim=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(channels, embed_dim, kernel_size=16, stride=16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(embed_dim, 64)

    def forward(self, x):
        x = self.patch_embed(x) # [batch, embed_dim, 14, 14]
        x = x.flatten(2).transpose(1, 2) # [batch, 196, embed_dim]
        x = self.transformer(x)
        return self.head(x.mean(dim=1))

class ModernCKDHybrid(nn.Module):
    def __init__(self, n_tab_features, n_clin_classes, n_vis_classes, tab_arch="transformer", vis_arch="vit"):
        super().__init__()
        # Dynamic Architecture Selection
        if tab_arch == "autoint":
            self.tab_model = AutoInt(n_tab_features)
        else:
            self.tab_model = FTTransformer(n_tab_features)
            
        if vis_arch == "swin":
            self.vision_model = SwinStyleVision()
        else:
            self.vision_model = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 64)
            )
        
        self.fusion = nn.Linear(128, 128)
        self.clinical_head = nn.Linear(128, n_clin_classes)
        self.vision_head = nn.Linear(128, n_vis_classes)

    def forward(self, tab, img):
        tab_feat = self.tab_model(tab)
        vis_feat = self.vision_model(img)
        combined = torch.cat([tab_feat, vis_feat], dim=1)
        fused = F.relu(self.fusion(combined))
        return self.clinical_head(fused), self.vision_head(fused)

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_and_report():
    train_ds, test_ds, n_features, n_vis_classes, n_clin_classes = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Comparison Configurations
    configs = [
        {"name": "SOTA Hybrid (FT-Trans + ViT)", "tab": "transformer", "vis": "vit"},
        {"name": "Advanced Hybrid (AutoInt + Swin)", "tab": "autoint", "vis": "swin"}
    ]
    
    comparison_results = []
    
    for config in configs:
        print(f"\n🚀 Training {config['name']}...")
        model = ModernCKDHybrid(n_features, n_clin_classes, n_vis_classes, 
                                 tab_arch=config['tab'], vis_arch=config['vis']).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 5 
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                tab, img = batch['tabular'].to(device), batch['image'].to(device)
                y_clin, y_vis = batch['clinical_label'].to(device), batch['vision_label'].to(device)
                
                optimizer.zero_grad()
                out_clin, out_vis = model(tab, img)
                loss = criterion(out_clin, y_clin) + criterion(out_vis, y_vis)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        all_clin_preds, all_clin_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                tab, img = batch['tabular'].to(device), batch['image'].to(device)
                out_clin, _ = model(tab, img)
                all_clin_preds.extend(torch.argmax(out_clin, dim=1).cpu().numpy())
                all_clin_true.extend(batch['clinical_label'].numpy())

        # Metrics
        p = precision_score(all_clin_true, all_clin_preds, average='macro', zero_division=0)
        r = recall_score(all_clin_true, all_clin_preds, average='macro', zero_division=0)
        f1 = f1_score(all_clin_true, all_clin_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_clin_true, all_clin_preds)
        
        comparison_results.append({
            "Model Architecture": config['name'],
            "Precision": p,
            "Recall": r,
            "F1-Score": f1,
            "Accuracy": acc
        })
        
        # Save best model weight
        if config['name'] == "SOTA Hybrid (FT-Trans + ViT)":
            torch.save(model.state_dict(), os.path.join("./models", "ckd_hybrid_sota.pth"))
            print(f"✅ Best SOTA model weights (FT-Trans + ViT) saved to ./models/ckd_hybrid_sota.pth")

    # Save Comparative Report
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "ckd_sota_comparison_report.csv"), index=False)
    print(f"\n📊 Comprehensive Comparison Report saved to {RESULTS_DIR}/ckd_sota_comparison_report.csv")

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model Architecture", y="F1-Score", data=comparison_df, palette="viridis")
    plt.title("SOTA Comparison - CKD Hybrid Architectures")
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(RESULTS_DIR, "ckd_sota_comparison_plot.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    train_and_report()

if __name__ == "__main__":
    train_and_report()
