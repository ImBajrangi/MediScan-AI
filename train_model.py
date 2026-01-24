
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

merged_path = "./datasets/merged_disease_symptoms.csv"
original_path = "./datasets/DiseaseAndSymptoms.csv"

if os.path.exists(merged_path):
    data_path = merged_path
    print(f"\n📂 Using MERGED dataset (expanded coverage)")
    is_merged = True
else:
    data_path = original_path
    print(f"\n📂 Using original dataset")
    is_merged = False

data = pd.read_csv(data_path)
print(f"   Loaded: {data_path}")

if is_merged:
    disease_col = 'Disease'
    symptom_cols = [col for col in data.columns if col != disease_col]
    all_symptoms = sorted(symptom_cols)
    X = data[symptom_cols].astype(int)
    le = LabelEncoder()
    y = le.fit_transform(data[disease_col])
else:
    cols = data.columns[1:]
    for col in cols:
        data[col] = data[col].str.strip()
    raw_symptoms = data[cols].values.ravel('K')
    all_symptoms = sorted(list(set([s for s in raw_symptoms if isinstance(s, str)])))
    X = pd.DataFrame(0, index=np.arange(len(data)), columns=all_symptoms)
    for i in range(len(data)):
        row_symptoms = data.iloc[i, 1:].dropna().values
        X.loc[i, row_symptoms] = 1
    le = LabelEncoder()
    y = le.fit_transform(data['Disease'])

from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler

print("\n🔄 Applying SMOTE oversampling for minority classes...")
class_counts = Counter(y)
min_samples = min(class_counts.values())
print(f"   Min samples per class: {min_samples}")
print(f"   Total classes: {len(class_counts)}")

target_samples = max(6, int(np.median(list(class_counts.values()))))
print(f"   Target samples per class: {target_samples}")

sampling_strategy = {}
for cls, count in class_counts.items():
    if count < target_samples:
        sampling_strategy[cls] = target_samples

if sampling_strategy:
    try:
        min_for_smote = min(class_counts.values())
        k_neighbors = min(5, min_for_smote - 1) if min_for_smote > 1 else 1
        if k_neighbors >= 1 and min_for_smote > 1:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"   ✓ SMOTE applied successfully")
        else:
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            print(f"   ✓ Random oversampling applied (some classes too small for SMOTE)")
        X = pd.DataFrame(X_resampled, columns=all_symptoms)
        y = np.array(y_resampled)
        print(f"   ✓ Dataset size: {len(y)} samples (was {sum(class_counts.values())})")
    except Exception as e:
        print(f"   ⚠️ Oversampling failed: {e}")
        print(f"   Using original data without oversampling")

final_class_counts = Counter(y)
print(f"   Final classes: {len(final_class_counts)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Dataset Statistics:")
print(f"   Total Samples: {len(X)}")
print(f"   Features (Symptoms): {len(all_symptoms)}")
print(f"   Classes (Diseases): {len(np.unique(y))}")
print(f"   Train Size: {len(X_train)}, Test Size: {len(X_test)}")
print(f"   Split Method: stratified")

large_dataset = len(X) > 10000
if large_dataset:
    print(f"\n⚡ Large dataset detected - using optimized parameters")
    print(f"   Estimated training time: 10-30 minutes")

print("\n🌲 Training Enhanced Random Forest...")
print("-" * 50)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"   ✓ Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"   ✓ Cross-Validation Score: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")

print("\n🚀 Training Enhanced XGBoost...")
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

print("\n🎯 Applying Probability Calibration...")
print("-" * 50)

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
    cv=2  # Use 2-fold for datasets with some classes having only 2 samples
)
calibrated_rf.fit(X_train, y_train)
y_pred_cal_rf = calibrated_rf.predict(X_test)
cal_rf_accuracy = accuracy_score(y_test, y_pred_cal_rf)
print(f"   ✓ Calibrated RF Accuracy: {cal_rf_accuracy * 100:.2f}%")

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
    cv=2  # Use 2-fold for datasets with some classes having only 2 samples
)
calibrated_xgb.fit(X_train, y_train)
y_pred_cal_xgb = calibrated_xgb.predict(X_test)
cal_xgb_accuracy = accuracy_score(y_test, y_pred_cal_xgb)
print(f"   ✓ Calibrated XGBoost Accuracy: {cal_xgb_accuracy * 100:.2f}%")

print("🔗 Creating MediScan Clinical Engine v2.0 (Voting)...")
print("="*50)

from sklearn.ensemble import VotingClassifier

    weights=[1.1, 0.9]
)

ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
ens_accuracy = accuracy_score(y_test, y_pred_ens)
probs_ens = ensemble.predict_proba(X_test)
avg_conf = np.mean(np.max(probs_ens, axis=1))

print(f"   ✓ Ensemble Accuracy: {ens_accuracy * 100:.2f}%")
print(f"   ✓ Average Prediction Confidence: {avg_conf * 100:.2f}%")
y_proba = ensemble.predict_proba(X_test)
avg_confidence = np.max(y_proba, axis=1).mean()
print(f"   ✓ Average Prediction Confidence: {avg_confidence * 100:.2f}%")

print("📦 Saving Models...")
print("=" * 60)

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

model_scores = {
    'Random Forest': rf_accuracy,
    'XGBoost': xgb_accuracy,
    'Calibrated RF': cal_rf_accuracy,
    'Calibrated XGBoost': cal_xgb_accuracy,
    'Ensemble': ens_accuracy
}

best_model_name = max(model_scores, key=model_scores.get)
best_accuracy = model_scores[best_model_name]

print(f"\n   Model Performance Summary:")
for name, score in model_scores.items():
    marker = "🏆" if name == best_model_name else "  "
    print(f"   {marker} {name}: {score * 100:.2f}%")

joblib.dump(ensemble, os.path.join(models_dir, "enhanced_disease_model.joblib"))
print(f"\n   ✓ Ensemble model saved")

joblib.dump(calibrated_rf, os.path.join(models_dir, "calibrated_rf_model.joblib"))
print(f"   ✓ Calibrated RF model saved")

joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
joblib.dump(all_symptoms, os.path.join(models_dir, "symptoms_list.joblib"))
print(f"   ✓ Label encoder and symptoms list saved")

print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n🎯 Best Model: {best_model_name} ({best_accuracy * 100:.2f}%)")
print(f"📊 Average Confidence: {avg_confidence * 100:.2f}%")
print(f"📁 Models saved to: {models_dir}/")
