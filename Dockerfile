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

# Download actual model files from Hugging Face Hub using the downloader script
RUN python3 hf_downloader.py

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 7860

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
