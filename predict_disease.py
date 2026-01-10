import pandas as pd
import numpy as np
import joblib
import os
import sys
import re

# Paths
models_dir = "./models"
datasets_dir = "./datasets"
model_path = os.path.join(models_dir, "best_disease_model_rf.joblib")
le_path = os.path.join(models_dir, "label_encoder.joblib")
symptoms_list_path = os.path.join(models_dir, "symptoms_list.joblib")
precaution_path = os.path.join(datasets_dir, "Disease precaution.csv")

# Load models and metadata
if not os.path.exists(model_path):
    print("Error: Models not found. Please run train_model.py first.")
    sys.exit(1)

model = joblib.load(model_path)
le = joblib.load(le_path)
all_symptoms = joblib.load(symptoms_list_path)

# Load precautions
precaution_df = pd.read_csv(precaution_path)
precaution_df['Disease'] = precaution_df['Disease'].str.strip()

def clean_symptom(s):
    if not isinstance(s, str): return None
    s = s.strip().lower()
    s = re.sub(r'[^a-zA-Z0-9\s_]', '', s)
    s = s.replace(' ', '_')
    return s

def predict_disease(user_symptoms):
    # Prepare input vector
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    
    # Clean and match symptoms
    matched_symptoms = []
    for s in user_symptoms:
        s_clean = clean_symptom(s)
        if s_clean in all_symptoms:
            input_vector.loc[0, s_clean] = 1
            matched_symptoms.append(s_clean)
    
    if not matched_symptoms:
        return None, 0.0, []
    
    # Predict
    prediction_idx = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    
    disease = le.inverse_transform([prediction_idx])[0]
    confidence = np.max(probabilities)
    
    return disease, confidence, matched_symptoms

if __name__ == "__main__":
    print("--- Disease Prediction System ---")
    print(f"Loaded {len(all_symptoms)} possible symptoms.")
    print("Enter symptoms separated by commas (e.g., itching, skin_rash, nodal_skin_eruptions):")
    
    user_input = input("> ")
    symptoms = [s.strip() for s in user_input.split(',')]
    
    result, score, matched = predict_disease(symptoms)
    
    print("\n--- Results ---")
    if matched:
        print(f"Matched symptoms: {', '.join(matched)}")
        print(f"Predicted Disease: {result}")
        print(f"Confidence: {score:.2%}")
        
        # Look up precautions
        precautions = precaution_df[precaution_df['Disease'] == result]
        if not precautions.empty:
            print("\nSuggested Precautions:")
            p_cols = [c for c in precautions.columns if 'Precaution' in c]
            for col in p_cols:
                p = precautions.iloc[0][col]
                if pd.notna(p) and str(p).strip():
                    print(f" - {p.strip().capitalize()}")
    else:
        print("No matching symptoms found in the database.")
