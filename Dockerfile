FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for downloads
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (excluding models which are LFS pointers)
COPY . .

# Download actual model files from Hugging Face LFS
RUN mkdir -p models && \
    curl -L -o models/best_disease_model_rf.joblib \
    "https://huggingface.co/spaces/mdark4025/MediScan-AI/resolve/main/models/best_disease_model_rf.joblib?download=true" && \
    curl -L -o models/label_encoder.joblib \
    "https://huggingface.co/spaces/mdark4025/MediScan-AI/resolve/main/models/label_encoder.joblib?download=true" && \
    curl -L -o models/symptoms_list.joblib \
    "https://huggingface.co/spaces/mdark4025/MediScan-AI/resolve/main/models/symptoms_list.joblib?download=true" && \
    curl -L -o models/vision_disease_model.pth \
    "https://huggingface.co/spaces/mdark4025/MediScan-AI/resolve/main/models/vision_disease_model.pth?download=true" && \
    curl -L -o models/vision_label_map.joblib \
    "https://huggingface.co/spaces/mdark4025/MediScan-AI/resolve/main/models/vision_label_map.joblib?download=true"

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 7860

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
