FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including git-lfs
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pull Git LFS files (model files)
RUN git lfs pull || echo "LFS pull skipped (not a git repo)"

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 7860

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
