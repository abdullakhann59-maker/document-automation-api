# ==============================================================================
# Document Automation API - Production Dockerfile
# ==============================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data

WORKDIR /app

# Install system dependencies for OpenCV, PyMuPDF, EasyOCR, and healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU build of PyTorch and torchvision first for optimal container size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, seed data, and utility scripts
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/seed_labels.json ./data/seed_labels.json

# Pre-create data directories
RUN mkdir -p /app/data/chroma /app/data/outputs /app/data/tmp

# Expose FastAPI service port
EXPOSE 8000

# Health check to monitor container readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
