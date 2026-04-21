FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller, faster)
RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Use Railway's PORT environment variable
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}