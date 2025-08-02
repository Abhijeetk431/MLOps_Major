# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Set PYTHONPATH to include src directory
ENV PYTHONPATH=/app/src

# Default command runs the prediction script
CMD ["python", "src/predict.py"]