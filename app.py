from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import joblib
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download

import json

app = Flask(__name__)
# Load Version Info
with open('version.json', 'r') as f:
    PROJ_VERSION = json.load(f)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
datasets_dir = os.path.join(BASE_DIR, "datasets")

# Hugging Face Repository Config
REPO_ID = "mdark4025/MediScan-AI"
REPO_TYPE = "space"

class CKDKidneyCNN(nn.Module):
    def __init__(self, n_classes=3):
        super(CKDKidneyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_path = os.path.join(models_dir, "vision_disease_model.pth")
label_map_path = os.path.join(models_dir, "vision_label_map.joblib")

def is_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer instead of actual content"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(50)
            return b'version https://git-lfs' in header
    except:
        return False

def get_model_file(filename):
    """Safely get or download a model file from Hugging Face Hub"""
    local_path = os.path.join(models_dir, filename)
    
    # Check if file exists and is not an LFS pointer
    if os.path.exists(local_path) and not is_lfs_pointer(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
        
    print(f"📥 Model {filename} missing or invalid. Downloading from HF...")
    try:
        download_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=f"models/{filename}",
            local_dir=BASE_DIR,
            local_dir_use_symlinks=False
        )
        return download_path
    except Exception as e:
        print(f"❌ Failed to sync {filename}: {e}")
        return local_path

def check_model_files():
    """Validate and sync all required model files"""
    os.makedirs(models_dir, exist_ok=True)
    
    assets = [
        "vision_disease_model.pth", 
        "vision_label_map.joblib",
        "enhanced_disease_model.joblib",
        "label_encoder.joblib",
        "symptoms_list.joblib",
        "ckd_clinical_model.joblib",
        "ckd_vision_model_enhanced.pth",
        "ckd_hybrid_sota.pth"
    ]
    
    for asset in assets:
        get_model_file(asset)

try:
    check_model_files()
    print("✓ All model files validated successfully")
except Exception as e:
    print(f"⚠ Model validation error: {e}")
    print("Starting in limited mode - some features may not work")

try:
    label_map = joblib.load(label_map_path)
    n_classes = len(label_map)
    model = SimpleCNN(n_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    vision_model_loaded = True
    print("✓ Vision model loaded successfully")
except Exception as e:
    print(f"⚠ Vision model failed to load: {e}")
    vision_model_loaded = False
    model = None
    label_map = {}

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

labels_pretty = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

symptom_model_path = os.path.join(models_dir, "calibrated_rf_model.joblib")
symptom_le_path = os.path.join(models_dir, "label_encoder.joblib")
symptoms_list_path = os.path.join(models_dir, "symptoms_list.joblib")
precaution_path = os.path.join(datasets_dir, "Disease precaution.csv")

try:
    if os.path.exists(symptom_model_path):
        symptom_model = joblib.load(symptom_model_path)
    else:
        # Fallback to ensemble if RF missing
        symptom_model = joblib.load(os.path.join(models_dir, "enhanced_disease_model.joblib"))
        
    symptom_le = joblib.load(symptom_le_path)
    all_symptoms = joblib.load(symptoms_list_path)
    precaution_df = pd.read_csv(precaution_path)
    precaution_df['Disease'] = precaution_df['Disease'].str.strip()
    symptom_model_loaded = True
    print("✓ Symptom model loaded successfully (Stable RF Mode)")
except Exception as e:
    print(f"⚠ Symptom model failed to load: {e}")
    symptom_model_loaded = False
    symptom_model = None
    symptom_le = None
    all_symptoms = []
# --- CKD Models ---
ckd_clinical_path = os.path.join(models_dir, "ckd_clinical_model.joblib")
ckd_vision_path = os.path.join(models_dir, "ckd_vision_model_enhanced.pth")

try:
    ckd_model = joblib.load(ckd_clinical_path)
    ckd_vision_model = CKDKidneyCNN(n_classes=3)
    ckd_vision_model.load_state_dict(torch.load(ckd_vision_path, map_location=torch.device('cpu')))
    ckd_vision_model.eval()
    ckd_models_loaded = True
    print("✓ CKD models loaded successfully")
except Exception as e:
    print(f"⚠ CKD models failed to load: {e}")
    ckd_models_loaded = False

import re
def clean_symptom(s):
    if not isinstance(s, str): return None
    s = s.strip().lower()
    s = re.sub(r'[^a-zA-Z0-9\s_]', '', s)
    s = s.replace(' ', '_')
    return s

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ckd')
def ckd_dashboard():
    return render_template('ckd_dashboard.html', 
                           version=PROJ_VERSION['app_version'],
                           env=PROJ_VERSION['environment'])

@app.route('/old_home')
def old_home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not vision_model_loaded or model is None:
        return jsonify({'error': 'Vision model not available. The model files may not have loaded correctly.'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = label_map[predicted.item()]
        pretty_name = labels_pretty.get(predicted_label, predicted_label)
        
        is_serious = predicted_label in ["mel", "bcc", "akiec"]
        
        return jsonify({
            'condition': pretty_name,
            'confidence': f"{confidence.item() * 100:.2f}",
            'is_serious': is_serious,
            'code': predicted_label
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    if not symptom_model_loaded or symptom_model is None:
        return jsonify({'error': 'Symptom model not available. The model files may not have loaded correctly.'}), 503
    
    data = request.json
    user_symptoms = data.get('symptoms', [])
    
    if not user_symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    try:
        input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
        
        matched_symptoms = []
        for s in user_symptoms:
            s_clean = clean_symptom(s)
            if s_clean in all_symptoms:
                input_vector.loc[0, s_clean] = 1
                matched_symptoms.append(s_clean)
        
        if not matched_symptoms:
            return jsonify({'error': 'No matching symptoms found in database'}), 404
        
        prediction_idx = symptom_model.predict(input_vector)[0]
        probabilities = symptom_model.predict_proba(input_vector)[0]
        
        disease = symptom_le.inverse_transform([prediction_idx])[0]
        confidence = np.max(probabilities)
        
        precautions = []
        prec_row = precaution_df[precaution_df['Disease'] == disease]
        if not prec_row.empty:
            p_cols = [c for c in precaution_df.columns if 'Precaution' in c]
            for col in p_cols:
                p = prec_row.iloc[0][col]
                if pd.notna(p) and str(p).strip():
                    precautions.append(p.strip().capitalize())
        
        return jsonify({
            'disease': disease,
            'confidence': f"{confidence * 100:.2f}",
            'precautions': precautions,
            'matched': matched_symptoms
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ckd', methods=['POST'])
def predict_ckd():
    if not ckd_models_loaded:
        return jsonify({'error': 'CKD models not loaded'}), 503
    
    data = request.json
    try:
        # Features: age, gfr, albuminuria
        features = pd.DataFrame([[
            data.get('age', 50),
            data.get('gfr', 60),
            data.get('albuminuria', 0)
        ]], columns=['age', 'gfr', 'albuminuria'])
        
        prediction = ckd_model.predict(features)[0]
        probabilities = ckd_model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        # Staging descriptions
        stage_desc = {
            "G1": "Stage 1: Normal or high GFR (>= 90)",
            "G2": "Stage 2: Mildly decreased GFR (60-89)",
            "G3a": "Stage 3a: Mildly to moderately decreased GFR (45-59)",
            "G3b": "Stage 3b: Moderately to severely decreased GFR (30-44)",
            "G4": "Stage 4: Severely decreased GFR (15-29)",
            "G5": "Stage 5: Kidney failure (< 15)"
        }
        
        return jsonify({
            'stage': prediction,
            'confidence': f"{confidence * 100:.2f}",
            'description': stage_desc.get(prediction, "Unknown Stage")
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ckd_vision', methods=['POST'])
def predict_ckd_vision():
    if not ckd_models_loaded:
        return jsonify({'error': 'CKD vision model not loaded'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = Image.open(filepath).convert('L').resize((28, 28))
        input_tensor = transforms.ToTensor()(image).unsqueeze(0)
        
        with torch.no_grad():
            output = ckd_vision_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        results = ["Normal", "Mild/Moderate CKD", "Severe CKD"]
        return jsonify({
            'status': results[predicted.item()],
            'confidence': f"{confidence.item() * 100:.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='127.0.0.1', port=port)
