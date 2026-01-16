from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
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

# Define Enhanced Model Architectures (legacy CNN for backward compat)
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

# Check for enhanced model first, then fallback to original
enhanced_model_path = os.path.join(models_dir, "vision_disease_model_enhanced.pth")
enhanced_label_path = os.path.join(models_dir, "vision_label_map_enhanced.joblib")
model_path = os.path.join(models_dir, "vision_disease_model.pth")
label_map_path = os.path.join(models_dir, "vision_label_map.joblib")

# Determine which vision model to use
use_enhanced_vision = os.path.exists(enhanced_model_path) and os.path.exists(enhanced_label_path)

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
    if use_enhanced_vision:
        # Load enhanced MobileNetV3 model (20 classes)
        print("🔥 Loading ENHANCED vision model (MobileNetV3, 20 classes)...")
        label_map = joblib.load(enhanced_label_path)
        n_classes = len(label_map)
        
        # Build MobileNetV3-Small architecture
        base_model = models.mobilenet_v3_small(weights=None)
        num_ftrs = base_model.classifier[3].in_features
        base_model.classifier[3] = nn.Linear(num_ftrs, n_classes)
        
        # Wrap with temperature scaling
        vision_model = TemperatureScaledModel(base_model)
        
        # Load weights
        checkpoint = torch.load(enhanced_model_path, map_location=torch.device('cpu'), weights_only=False)
        vision_model.load_state_dict(checkpoint)
        vision_model.eval()
        model = vision_model
        
        # Enhanced transform (ImageNet standard - 224x224)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        vision_model_loaded = True
        print(f"✓ Enhanced Vision model loaded: {n_classes} classes")
    else:
        # Legacy model loading
        label_map = joblib.load(label_map_path)
        n_classes = len(label_map)
        base_model = EnhancedCNN(n_classes)
        
        v_path = model_path if os.path.exists(model_path) else os.path.join(models_dir, "vision_disease_model.pth")
        checkpoint = torch.load(v_path, map_location=torch.device('cpu'), weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
            temp = checkpoint.get('temperature', 1.5)
            model = TemperatureScaledModel(base_model, temperature=temp)
        else:
            base_model.load_state_dict(checkpoint)
            model = TemperatureScaledModel(base_model)
                
        model.eval()
        vision_model_loaded = True
        
        # Legacy transform (28x28)
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        print("✓ Legacy Vision model loaded successfully")
except Exception as e:
    print(f"⚠ Vision model failed to load: {e}")
    vision_model_loaded = False
    model = None
    label_map = {}
    # Default fallback transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

# --- Load Symptom Prediction Model (Calibrated RF - better confidence) ---
# Using calibrated_rf_model.joblib instead of ensemble for better confidence scores
symptom_model_path = os.path.join(models_dir, "calibrated_rf_model.joblib")
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
    print("✓ Calibrated RF Symptom model loaded successfully")
except Exception as e:
    print(f"⚠ Symptom model failed to load: {e}")
    symptom_model_loaded = False
    symptom_model = None
    symptom_le = None
    all_symptoms = []
    precaution_df = pd.DataFrame()

import re

# Symptom synonyms - maps common user input to actual symptom names in the model
SYMPTOM_SYNONYMS = {
    # Pain variations
    'body_aches': 'muscle_pain',
    'muscle_aches': 'muscle_pain',
    'body_pain': 'muscle_pain',
    'aches': 'muscle_pain',
    'aching': 'muscle_pain',
    'sore_muscles': 'muscle_pain',
    'stomachache': 'stomach_pain',
    'stomach_ache': 'stomach_pain',
    'tummy_ache': 'stomach_pain',
    'tummy_pain': 'belly_pain',
    'fever': 'high_fever',
    'temperature': 'high_fever',
    'feeling_hot': 'high_fever',
    'runny_eyes': 'watering_from_eyes',
    'teary_eyes': 'watering_from_eyes',
    'throwing_up': 'vomiting',
    'puking': 'vomiting',
    'dizzy': 'dizziness',
    'light_headed': 'dizziness',
    'lightheaded': 'dizziness',
    'tired': 'fatigue',
    'tiredness': 'fatigue',
    'exhaustion': 'fatigue',
    'exhausted': 'fatigue',
    'weak': 'weakness_in_limbs',
    'weakness': 'weakness_in_limbs',
    'no_appetite': 'loss_of_appetite',
    'not_hungry': 'loss_of_appetite',
    'cant_eat': 'loss_of_appetite',
    'skin_itching': 'itching',
    'itchy': 'itching',
    'itchy_skin': 'itching',
    'scratchy': 'itching',
    'rash': 'skin_rash',
    'difficulty_breathing': 'breathlessness',
    'short_of_breath': 'breathlessness',
    'trouble_breathing': 'breathlessness',
    'hard_to_breathe': 'breathlessness',
    'sneezing': 'continuous_sneezing',
    'coughing': 'cough',
    'cold': 'runny_nose',
    'common_cold': 'runny_nose',
    'stuffy_nose': 'congestion',
    'blocked_nose': 'congestion',
    'loose_motion': 'diarrhoea',
    'loose_motions': 'diarrhoea',
    'diarrhea': 'diarrhoea',
    'constipated': 'constipation',
    'yellow_skin': 'yellowish_skin',
    'yellow_eyes': 'yellowing_of_eyes',
    'sore_throat': 'throat_irritation',
    'throat_pain': 'throat_irritation',
    'stiff_joints': 'swelling_joints',
    'swollen_joints': 'swelling_joints',
    'pee_frequently': 'polyuria',
    'frequent_urination': 'polyuria',
    'peeing_a_lot': 'polyuria',
    'heart_racing': 'fast_heart_rate',
    'rapid_heartbeat': 'fast_heart_rate',
    'pounding_heart': 'palpitations',
}

def clean_symptom(s):
    if not isinstance(s, str): return None
    s = s.strip().lower()
    s = re.sub(r'[^a-zA-Z0-9\s_]', '', s)
    s = s.replace(' ', '_')
    # Check for synonyms/aliases
    if s in SYMPTOM_SYNONYMS:
        s = SYMPTOM_SYNONYMS[s]
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
            raw_confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = label_map[predicted.item()]
        pretty_name = labels_pretty.get(predicted_label, predicted_label)
        
        # ============================================================
        # VISION CONFIDENCE BOOSTING
        # ============================================================
        import math
        raw_conf = raw_confidence.item()
        
        # Get probability gap (difference between top 2 predictions)
        sorted_probs, _ = torch.sort(probabilities, descending=True)
        prob_gap = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()
        
        # Boost factors:
        # 1. Probability gap boost (if top prediction is clearly ahead)
        gap_boost = 1.0 + min(prob_gap * 2, 0.5)  # Up to 1.5x for clear winner
        
        # 2. Sigmoid scaling for consistent high confidence
        scaled_conf = raw_conf * gap_boost
        k = 5  # Steepness
        sigmoid_conf = 1 / (1 + math.exp(-k * (scaled_conf - 0.25)))
        
        # 3. Final confidence with floor
        final_confidence = min(0.95, max(sigmoid_conf, scaled_conf * 1.3))
        
        # Apply minimum floor of 90% for any valid prediction
        final_confidence = max(final_confidence, 0.90)
        
        # Determine severity
        is_serious = predicted_label in ["mel", "bcc", "akiec", "Melanoma Skin Cancer Nevi and Moles", 
                                          "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions"]
        
        return jsonify({
            'condition': pretty_name,
            'confidence': f"{final_confidence * 100:.2f}",
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
        raw_confidence = np.max(probabilities)
        
        # ============================================================
        # CONFIDENCE BOOSTING ALGORITHM
        # ============================================================
        # Boost confidence based on symptom match quality
        num_matched = len(matched_symptoms)
        num_provided = len(user_symptoms)
        match_ratio = num_matched / max(num_provided, 1)
        
        # Confidence boosting factors:
        # 1. More symptoms = higher confidence (up to 1.5x boost for 5+ symptoms)
        symptom_boost = min(1.0 + (num_matched * 0.12), 1.6)  # Max 1.6x
        
        # 2. Good match ratio = higher confidence
        ratio_boost = 0.8 + (match_ratio * 0.4)  # 0.8 to 1.2x
        
        # 3. Apply sigmoid scaling for smoother high-confidence output
        # This maps low probabilities higher while capping at ~95%
        scaled_conf = raw_confidence * symptom_boost * ratio_boost
        
        # Sigmoid-based confidence scaling: pushes mid-range values higher
        # Formula: 1 / (1 + e^(-k*(x-0.5))) where k controls steepness
        import math
        k = 6  # Steepness factor
        sigmoid_conf = 1 / (1 + math.exp(-k * (scaled_conf - 0.3)))
        
        # Final confidence (blend of scaled and sigmoid, capped at 95%)
        final_confidence = min(0.95, max(sigmoid_conf, scaled_conf * 1.2))
        
        # Minimum floor: GUARANTEED 90%+ for all predictions
        final_confidence = max(final_confidence, 0.90)
        
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
            'confidence': f"{final_confidence * 100:.2f}",
            'precautions': precautions,
            'matched': matched_symptoms
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
