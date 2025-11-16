# Base image Python
FROM python:3.12.12-slim

# Working directory
WORKDIR /app

# Copy req.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy best model
COPY best_model.pkl ./model.pkl
COPY serve.py .

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*