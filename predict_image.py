import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import joblib
import os
import sys

# Define the same model architecture as used in training
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

# Paths
models_dir = "./models"
model_path = os.path.join(models_dir, "vision_disease_model.pth")
label_map_path = os.path.join(models_dir, "vision_label_map.joblib")

def predict_image(image_path):
    if not os.path.exists(model_path):
        print("Error: Vision model not found. Please run train_vision_model.py first.")
        return

    # Load metadata
    label_map = joblib.load(label_map_path)
    n_classes = len(label_map)

    # Initialize model
    model = SimpleCNN(n_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = label_map[predicted.item()]
    
    # Prettify labels
    labels_pretty = {
        "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis-like Lesions",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi",
        "vasc": "Vascular Lesions"
    }
    
    pretty_name = labels_pretty.get(predicted_label, predicted_label)

    print("\n--- Pictorial Diagnosis Results ---")
    print(f"Detected Condition: {pretty_name}")
    print(f"Confidence: {confidence.item():.2%}")
    
    if predicted_label in ["mel", "bcc", "akiec"]:
        print("\n[WARNING] This condition may require urgent medical attention.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_image.py <path_to_image>")
        sys.exit(1)
    
    predict_image(sys.argv[1])
