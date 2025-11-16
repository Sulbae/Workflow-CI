# Base image Python
FROM python:3.12.12-slim

# Working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy req.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy best model
COPY model/best_model ./model/best_model

# Copy inference script
COPY inference.py .

# Expose API port (FastAPI/Flask)
EXPOSE 8080

# 