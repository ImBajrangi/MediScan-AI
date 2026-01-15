"""
MediScan AI - Enhanced Disease Prediction Model Training
Features: Hyperparameter tuning, Probability Calibration, Ensemble Voting
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MediScan AI - Enhanced Symptom-Based Model Training")
print("=" * 60)

# Load the dataset
data_path = "./datasets/DiseaseAndSymptoms.csv"
data = pd.read_csv(data_path)

# 1. Preprocessing
cols = data.columns[1:]
for col in cols:
    data[col] = data[col].str.strip()

# Get unique symptoms and filter out NaN
raw_symptoms = data[cols].values.ravel('K')
all_symptoms = sorted(list(set([s for s in raw_symptoms if isinstance(s, str)])))

# Create binary matrix (X)
X = pd.DataFrame(0, index=np.arange(len(data)), columns=all_symptoms)
for i in range(len(data)):
    row_symptoms = data.iloc[i, 1:].dropna().values
    X.loc[i, row_symptoms] = 1

# Encode labels (y)
le = LabelEncoder()
y = le.fit_transform(data['Disease'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Dataset Statistics:")
print(f"   Total Samples: {len(data)}")
print(f"   Features (Symptoms): {len(all_symptoms)}")
print(f"   Classes (Diseases): {len(le.classes_)}")
print(f"   Train Size: {len(X_train)}, Test Size: {len(X_test)}")

# ============================================================================
# ENHANCED RANDOM FOREST
# ============================================================================
print("\n" + "-" * 50)
print("🌲 Training Enhanced Random Forest...")
print("-" * 50)

rf = RandomForestClassifier(
    n_estimators=500,           # Increased from 100
    max_depth=25,               # Limit depth to prevent overfitting
    min_samples_split=5,        # Minimum samples to split
    min_samples_leaf=2,         # Minimum samples in leaf
    class_weight='balanced',    # Handle class imbalance
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"   ✓ Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# Cross-validation score
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"   ✓ Cross-Validation Score: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")

# ============================================================================
# ENHANCED XGBOOST
# ============================================================================
print("\n" + "-" * 50)
print("🚀 Training Enhanced XGBoost...")
print("-" * 50)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=15,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"   ✓ XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")

# ============================================================================
# PROBABILITY CALIBRATION
# ============================================================================
print("\n" + "-" * 50)
print("🎯 Applying Probability Calibration...")
print("-" * 50)

# Calibrate RandomForest
print("   Calibrating Random Forest (this may take a few minutes)...")
calibrated_rf = CalibratedClassifierCV(
    estimator=RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    method='isotonic',
    cv=3
)
calibrated_rf.fit(X_train, y_train)
y_pred_cal_rf = calibrated_rf.predict(X_test)
cal_rf_accuracy = accuracy_score(y_test, y_pred_cal_rf)
print(f"   ✓ Calibrated RF Accuracy: {cal_rf_accuracy * 100:.2f}%")

# Calibrate XGBoost
print("   Calibrating XGBoost...")
calibrated_xgb = CalibratedClassifierCV(
    estimator=xgb.XGBClassifier(
        n_estimators=200,
        max_depth=12,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    ),
    method='isotonic',
    cv=3
)
calibrated_xgb.fit(X_train, y_train)
y_pred_cal_xgb = calibrated_xgb.predict(X_test)
cal_xgb_accuracy = accuracy_score(y_test, y_pred_cal_xgb)
print(f"   ✓ Calibrated XGBoost Accuracy: {cal_xgb_accuracy * 100:.2f}%")

# ============================================================================
# ENSEMBLE WITH SOFT VOTING
# ============================================================================
print("\n" + "-" * 50)
print("🔗 Creating Ensemble with Soft Voting...")
print("-" * 50)

# Create fresh models for ensemble
rf_for_ensemble = RandomForestClassifier(
    n_estimators=400,
    max_depth=22,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

xgb_for_ensemble = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=14,
    learning_rate=0.1,
    subsample=0.85,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_for_ensemble),
        ('xgb', xgb_for_ensemble)
    ],
    voting='soft',  # Use probability averaging for better confidence
    weights=[1.2, 1.0]  # Slightly favor RF if it performs better
)

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"   ✓ Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")

# Test confidence scores
y_proba = ensemble.predict_proba(X_test)
avg_confidence = np.max(y_proba, axis=1).mean()
print(f"   ✓ Average Prediction Confidence: {avg_confidence * 100:.2f}%")

# ============================================================================
# SELECT AND SAVE BEST MODEL
# ============================================================================
print("\n" + "=" * 60)
print("📦 Saving Models...")
print("=" * 60)

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

# Compare all models
model_scores = {
    'Random Forest': rf_accuracy,
    'XGBoost': xgb_accuracy,
    'Calibrated RF': cal_rf_accuracy,
    'Calibrated XGBoost': cal_xgb_accuracy,
    'Ensemble': ensemble_accuracy
}

best_model_name = max(model_scores, key=model_scores.get)
best_accuracy = model_scores[best_model_name]

print(f"\n   Model Performance Summary:")
for name, score in model_scores.items():
    marker = "🏆" if name == best_model_name else "  "
    print(f"   {marker} {name}: {score * 100:.2f}%")

# Save ensemble as primary model (best for confidence)
joblib.dump(ensemble, os.path.join(models_dir, "enhanced_disease_model.joblib"))
print(f"\n   ✓ Ensemble model saved")

# Also save calibrated RF as backup
joblib.dump(calibrated_rf, os.path.join(models_dir, "calibrated_rf_model.joblib"))
print(f"   ✓ Calibrated RF model saved")

# Save supporting files
joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
joblib.dump(all_symptoms, os.path.join(models_dir, "symptoms_list.joblib"))
print(f"   ✓ Label encoder and symptoms list saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n🎯 Best Model: {best_model_name} ({best_accuracy * 100:.2f}%)")
print(f"📊 Average Confidence: {avg_confidence * 100:.2f}%")
print(f"📁 Models saved to: {models_dir}/")
