# 🏥 MediScan AI - Disease Prediction System

A comprehensive AI-powered disease prediction platform that combines **symptom-based analysis** and **image-based diagnosis** for skin conditions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🔬 Dual Diagnosis System
- **Symptom Analysis**: Enter symptoms in natural language and get AI-powered disease predictions
- **Pictorial Diagnosis**: Upload skin lesion images for computer vision-based analysis

### 🎨 Modern Web Interface
- Glassmorphism design with dark theme
- Responsive layout for all devices
- Real-time analysis with loading states
- Professional PDF report generation

### 📊 AI Models
| Model | Type | Coverage | Accuracy |
|-------|------|----------|----------|
| Random Forest | Symptom-based | 426 diseases | 93.25% |
| SimpleCNN | Image-based | 7 skin conditions | 73.12% |

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/MediScan-AI.git
   cd MediScan-AI
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask torch torchvision pandas scikit-learn joblib pillow numpy
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5001`

## 📁 Project Structure

```
MediScan-AI/
├── app.py                  # Flask web application
├── train_model.py          # Symptom model training
├── train_vision_model.py   # Vision model training
├── predict_disease.py      # CLI symptom prediction
├── predict_image.py        # CLI image prediction
├── models/                 # Trained AI models
│   ├── best_disease_model_rf.joblib
│   ├── vision_disease_model.pth
│   └── ...
├── datasets/               # Training data
├── static/                 # CSS & JavaScript
├── templates/              # HTML templates
└── samples/                # Test images
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Image-based diagnosis |
| `/predict_symptoms` | POST | Symptom-based diagnosis |

## 📱 Screenshots

The application features a modern, dark-themed interface with:
- Tab-based navigation between diagnosis modes
- Drag-and-drop image upload
- Real-time results display
- PDF report download

## ⚠️ Medical Disclaimer

This system uses artificial intelligence for preliminary analysis and is **NOT a substitute for professional medical diagnosis**. Always consult a qualified healthcare professional for medical decisions.

## 🛠️ Technologies Used

- **Backend**: Flask, Python
- **AI/ML**: PyTorch, scikit-learn, Random Forest
- **Frontend**: HTML5, CSS3, JavaScript
- **PDF Generation**: jsPDF
- **Dataset**: DermaMNIST (skin lesions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👨‍💻 Author

Created with ❤️ using AI-assisted development.

---

**Note**: The trained models are included in the `models/` directory. If you want to retrain, run `train_model.py` and `train_vision_model.py`.
