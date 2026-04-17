import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier

RESULTS_DIR = "./evaluation_results/ckd_research"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('dark_background')
sns.set_palette(['#00d2ff', '#50fa7b', '#8be9fd', '#ff79c6', '#bd93f9'])
plt.rcParams.update({
    "figure.facecolor": "#121212",
    "axes.facecolor": "#121212",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "grid.color": "#2c2c2c",
    "font.family": "sans-serif",
    "font.size": 10
})

def process_research_data(file_path):
    print(f"🔍 Loading dataset: {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    
    columns_to_keep = ['age', 'sc', 'al', 'bp', 'su', 'class']
    df = df[columns_to_keep].copy()

    for col in ['age', 'sc', 'al', 'bp', 'su']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    df['gfr'] = 186 * (df['sc'] ** -1.154) * (df['age'] ** -0.203)
    
    df['class'] = df['class'].astype(str).str.strip().str.lower()

    def determine_stage(row):
        g = row['gfr']
        if 'notckd' in row['class']:
            return "Normal"
        if g >= 90: return "G1"
        if g >= 60: return "G2"
        if g >= 45: return "G3a"
        if g >= 30: return "G3b"
        if g >= 15: return "G4"
        return "G5"
    
    df['stage'] = df.apply(determine_stage, axis=1)
    
    return df[['age', 'gfr', 'al', 'bp', 'su', 'stage']]

data_path = "datasets/kidney_disease_raw.csv"
if not os.path.exists(data_path):
    print(f"❌ Error: {data_path} not found.")
    exit(1)

processed_df = process_research_data(data_path)
X = processed_df[['age', 'gfr', 'al', 'bp', 'su']]
y = processed_df['stage']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
n_classes = len(class_names)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🏋️ Training models...")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
}

hybrid_model = VotingClassifier(
    estimators=[
        ('rf', models["Random Forest"]),
        ('svm', models["SVM"]),
        ('gb', models["Gradient Boosting"])
    ],
    voting='soft'
)
models["Hybrid (Ensemble)"] = hybrid_model

performance_metrics = []
model_outputs = {}

for name, model in models.items():
    print(f"   ➤ Training {name}...")
    if "SVM" in name or "Hybrid" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except:
        auc_score = 0.0
        
    performance_metrics.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc_score
    })
    
    model_outputs[name] = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

metrics_df = pd.DataFrame(performance_metrics)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "ckd_research_metrics.csv"), index=False)
print(f"✅ Metrics saved to {RESULTS_DIR}/ckd_research_metrics.csv")

print("📊 Generating Visualizations...")

fig, ax = plt.subplots(figsize=(12, 7))
tidy = metrics_df.melt(id_vars='Model').rename(columns=str.title)
sns.barplot(x='Model', y='Value', hue='Variable', data=tidy, ax=ax, palette='viridis')
ax.set_title("CKD Model Performance Comparison", fontsize=16, pad=20)
ax.set_ylim(0, 1.05)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ckd_comparison_metrics.png"), dpi=200)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
for i, (name, output) in enumerate(model_outputs.items()):
    cm = confusion_matrix(output['y_test'], output['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[i].set_title(f"Confusion Matrix: {name}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ckd_confusion_matrices.png"), dpi=200)
plt.close()

plt.figure(figsize=(10, 8))
y_test_bin = pd.get_dummies(y_test).values
y_proba_hybrid = model_outputs["Hybrid (Ensemble)"]["y_proba"]

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_hybrid[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves (Hybrid Ensemble Model)', fontsize=14)
plt.legend(loc="lower right")
plt.savefig(os.path.join(RESULTS_DIR, "ckd_roc_curves.png"), dpi=200)
plt.close()

print(f"✨ Analysis complete. See results in {RESULTS_DIR}")
