import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate Synthetic CKD Clinical Data
# CKD Staging based on GFR:
# G1: GFR >= 90
# G2: GFR 60-89
# G3a: GFR 45-59
# G3b: GFR 30-44
# G4: GFR 15-29
# G5: GFR < 15

def generate_synthetic_ckd_data(n_samples=1000):
    np.random.seed(42)
    age = np.random.randint(18, 90, n_samples)
    gfr = np.random.randint(5, 120, n_samples)
    albuminuria = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]) # 0: Normal, 1: Mod inc, 2: Sev inc
    
    # Simple labels based on GFR (primary indicator)
    stages = []
    for g in gfr:
        if g >= 90: stages.append("G1")
        elif g >= 60: stages.append("G2")
        elif g >= 45: stages.append("G3a")
        elif g >= 30: stages.append("G3b")
        elif g >= 15: stages.append("G4")
        else: stages.append("G5")
    
    df = pd.DataFrame({
        'age': age,
        'gfr': gfr,
        'albuminuria': albuminuria,
        'stage': stages
    })
    return df

print("Generating synthetic CKD clinical data...")
data = generate_synthetic_ckd_data()

# 2. Train Model
X = data[['age', 'gfr', 'albuminuria']]
y = data['stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 3. Save Model
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, "ckd_clinical_model.joblib"))
print(f"Model saved to {models_dir}/ckd_clinical_model.joblib")
