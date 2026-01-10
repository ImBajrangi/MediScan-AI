import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data_path = "./datasets/DiseaseAndSymptoms.csv"
data = pd.read_csv(data_path)

# 1. Preprocessing
# Clean whitespace
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

print(f"X_train shape: {X_train.shape}")

# 2. Random Forest
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# 3. XGBoost
print("\n--- Training XGBoost ---")
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

# Save the best model (comparing accuracies)
models_dir = "/Users/mr.bajrangi/Data Science Project/models"
os.makedirs(models_dir, exist_ok=True)

if accuracy_score(y_test, y_pred_rf) >= accuracy_score(y_test, y_pred_xgb):
    print("\nSaving Random Forest as the best model.")
    best_model = rf
    model_name = "best_disease_model_rf.joblib"
else:
    print("\nSaving XGBoost as the best model.")
    best_model = xgb_model
    model_name = "best_disease_model_xgb.joblib"

joblib.dump(best_model, os.path.join(models_dir, model_name))
joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
joblib.dump(all_symptoms, os.path.join(models_dir, "symptoms_list.joblib"))

print("\nModel training completed and saved to:", models_dir)
