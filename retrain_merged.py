import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import re

# Paths
datasets_dir = "./datasets"
ds1_path = os.path.join(datasets_dir, "DiseaseAndSymptoms.csv")
ds2_path = os.path.join(datasets_dir, "Diseases_Symptoms.csv")
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

def clean_symptom(s):
    if not isinstance(s, str): return None
    s = s.strip().lower()
    s = re.sub(r'[^a-zA-Z0-9\s_]', '', s)
    s = s.replace(' ', '_')
    return s

# 1. Load and Process Dataset 1 (Columnar)
print("Processing Dataset 1...")
df1 = pd.read_csv(ds1_path)
symptom_cols = [c for c in df1.columns if 'Symptom' in c]
ds1_processed = []

for _, row in df1.iterrows():
    symptoms = [clean_symptom(row[c]) for c in symptom_cols if pd.notna(row[c])]
    symptoms = [s for s in symptoms if s]
    ds1_processed.append({'Disease': row['Disease'].strip(), 'Symptoms': list(set(symptoms))})

# 2. Load and Process Dataset 2 (Comma-separated string)
print("Processing Dataset 2...")
df2 = pd.read_csv(ds2_path)
ds2_processed = []

for _, row in df2.iterrows():
    if pd.isna(row['Symptoms']): continue
    # Split by comma and clean
    raw_symptoms = row['Symptoms'].split(',')
    symptoms = [clean_symptom(s) for s in raw_symptoms]
    symptoms = [s for s in symptoms if s]
    # We might only have 1 row per disease here, so we might want to duplicate or just use it
    ds2_processed.append({'Disease': row['Name'].strip(), 'Symptoms': list(set(symptoms))})

# Combine
combined_data = ds1_processed + ds2_processed
all_diseases = [d['Disease'] for d in combined_data]
all_symptoms_list = []
for d in combined_data:
    all_symptoms_list.extend(d['Symptoms'])
unique_symptoms = sorted(list(set(all_symptoms_list)))

print(f"Total Diseases: {len(set(all_diseases))}")
print(f"Total Unique Symptoms: {len(unique_symptoms)}")

# 3. Create Binary Matrix
print("Building feature matrix...")
X = pd.DataFrame(0, index=np.arange(len(combined_data)), columns=unique_symptoms)
for i, entry in enumerate(combined_data):
    for sym in entry['Symptoms']:
        X.loc[i, sym] = 1

le = LabelEncoder()
y = le.fit_transform(all_diseases)

# 4. Train Model
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 5. Save
joblib.dump(rf, os.path.join(models_dir, "best_disease_model_rf.joblib"))
joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
joblib.dump(unique_symptoms, os.path.join(models_dir, "symptoms_list.joblib"))

print("\nRetraining completed with merged datasets.")
