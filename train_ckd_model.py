import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and Process Real UCI CKD Data
def process_real_data(file_path):
    # Load dataset (ignoring the 'id' column)
    df = pd.read_csv(file_path, index_col=0)
    
    # Clean numeric columns and handle missing values
    columns_to_keep = ['age', 'sc', 'al', 'class']
    df = df[columns_to_keep].copy()
    
    # Convert to numeric, errors='coerce' turns non-numeric strings into NaN
    for col in ['age', 'sc', 'al']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Simple imputation: fill NaNs with median
    df['age'] = df['age'].fillna(df['age'].median())
    df['sc'] = df['sc'].fillna(df['sc'].median())
    df['al'] = df['al'].fillna(df['al'].median())
    
    # Calculate GFR (MDRD Simplified Formula)
    # GFR = 186 * (SC^-1.154) * (Age^-0.203)
    # Note: This is an approximation as we don't have sex/race for every row
    df['gfr'] = 186 * (df['sc'] ** -1.154) * (df['age'] ** -0.203)
    
    # Clean class column (some rows have 'ckd\t' or 'no')
    df['class'] = df['class'].str.strip().str.lower()
    
    # Define Stages based on GFR (Standard Clinical Staging)
    def get_stage(row):
        g = row['gfr']
        if row['class'] != 'ckd':
            return "G1" # Normal
        if g >= 90: return "G1"
        if g >= 60: return "G2"
        if g >= 45: return "G3a"
        if g >= 30: return "G3b"
        if g >= 15: return "G4"
        return "G5"
    
    df['stage'] = df.apply(get_stage, axis=1)
    
    # Map Albuminuria to UI scale (0, 1, 2)
    # UCI 'al' is 0-5. Map 0 -> 0, 1-2 -> 1, 3-5 -> 2
    def map_al(al):
        if al == 0: return 0
        if al <= 2: return 1
        return 2
    df['al_mapped'] = df['al'].apply(map_al)
    
    return df[['age', 'gfr', 'al_mapped', 'stage']]

print("Processing real UCI CKD dataset...")
data_path = "datasets/kidney_disease_raw.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
else:
    processed_df = process_real_data(data_path)
    
    # 2. Train Model
    X = processed_df[['age', 'gfr', 'al_mapped']]
    X.columns = ['age', 'gfr', 'albuminuria'] # Match UI expected names
    y = processed_df['stage']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Random Forest for fast and robust training
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Real-Data Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 3. Save Model
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "ckd_clinical_model.joblib"))
    print(f"Optimized model saved to {models_dir}/ckd_clinical_model.joblib")
