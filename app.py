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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Base directory for paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
datasets_dir = os.path.join(BASE_DIR, "datasets")

# Define Enhanced Model Architectures
class EnhancedCNN(nn.Module):
    def __init__(self, n_classes):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout2d(0.25)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.dropout_conv(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout_conv(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout_conv(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Temperature Scaling Wrapper
class TemperatureScaledModel(nn.Module):
    def __init__(self, model, temperature=1.5):
        super(TemperatureScaledModel, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

# Load models
model_path = os.path.join(models_dir, "best_vision_model.pth")
label_map_path = os.path.join(models_dir, "vision_label_map.joblib")

# Check if model files are Git LFS pointers (not actual files)
def is_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer instead of actual content"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(50)
            return b'version https://git-lfs' in header
    except:
        return False

def check_model_files():
    """Validate all model files are properly downloaded"""
    model_files = [
        model_path, 
        label_map_path,
        os.path.join(models_dir, "enhanced_disease_model.joblib"),
        os.path.join(models_dir, "label_encoder.joblib"),
        os.path.join(models_dir, "symptoms_list.joblib")
    ]
    for mf in model_files:
        if not os.path.exists(mf):
            # Fallback for vision model name
            if mf == model_path and os.path.exists(os.path.join(models_dir, "vision_disease_model.pth")):
                continue
            raise FileNotFoundError(f"Model file not found: {mf}")
        if is_lfs_pointer(mf):
            raise ValueError(f"Git LFS file not downloaded properly: {mf}. "
                           f"Run 'git lfs pull' to download model files.")

# Try to load models with error handling
try:
    label_map = joblib.load(label_map_path)
    n_classes = len(label_map)
    base_model = EnhancedCNN(n_classes)
    
    # Check which vision model file to use
    v_path = model_path if os.path.exists(model_path) else os.path.join(models_dir, "vision_disease_model.pth")
    
    checkpoint = torch.load(v_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Handle both full state_dict and just weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
        temp = checkpoint.get('temperature', 1.5)
        model = TemperatureScaledModel(base_model, temperature=temp)
    else:
        # Fallback for simple state_dict
        try:
            base_model.load_state_dict(checkpoint)
            model = TemperatureScaledModel(base_model)
        except:
            # If architecture mismatch (old model), try loading as SimpleCNN
            from train_vision_model import SimpleCNN # If available
            # For now, we assume the user is using the new one
            raise
            
    model.eval()
    vision_model_loaded = True
    print("✓ Vision model loaded successfully")
except Exception as e:
    print(f"⚠ Vision model failed to load: {e}")
    vision_model_loaded = False
    model = None
    label_map = {}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# Label mapping for vision
labels_pretty = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

# --- Load Symptom Prediction Model (Ensemble) ---
symptom_model_path = os.path.join(models_dir, "enhanced_disease_model.joblib")
symptom_le_path = os.path.join(models_dir, "label_encoder.joblib")
symptoms_list_path = os.path.join(models_dir, "symptoms_list.joblib")
precaution_path = os.path.join(datasets_dir, "Disease precaution.csv")

try:
    symptom_model = joblib.load(symptom_model_path)
    symptom_le = joblib.load(symptom_le_path)
    all_symptoms = joblib.load(symptoms_list_path)
    precaution_df = pd.read_csv(precaution_path)
    precaution_df['Disease'] = precaution_df['Disease'].str.strip()
    symptom_model_loaded = True
    print("✓ Enhanced Symptom model loaded successfully")
except Exception as e:
    print(f"⚠ Symptom model failed to load: {e}")
    symptom_model_loaded = False
    symptom_model = None
    symptom_le = None
    all_symptoms = []
    precaution_df = pd.DataFrame()

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

@app.route('/predict', methods=['POST'])
def predict():
    # Check if vision model is available
    if not vision_model_loaded or model is None:
        return jsonify({'error': 'Vision model not available. The model files may not have loaded correctly.'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and transform image
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = label_map[predicted.item()]
        pretty_name = labels_pretty.get(predicted_label, predicted_label)
        
        # Determine severity
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
    # Check if symptom model is available
    if not symptom_model_loaded or symptom_model is None:
        return jsonify({'error': 'Symptom model not available. The model files may not have loaded correctly.'}), 503
    
    data = request.json
    user_symptoms = data.get('symptoms', [])
    
    if not user_symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    try:
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
            return jsonify({'error': 'No matching symptoms found in database'}), 404
        
        # Predict
        prediction_idx = symptom_model.predict(input_vector)[0]
        probabilities = symptom_model.predict_proba(input_vector)[0]
        
        disease = symptom_le.inverse_transform([prediction_idx])[0]
        confidence = np.max(probabilities)
        
        # Look up precautions
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
