FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLOUDS_API_ENV=Production \
    FLOUDS_DEBUG_MODE=0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY app ./app
#COPY onnx_loaders ./onnx_loaders

# Expose the port (default: 5001)
EXPOSE 5001

# Set ONNX model directory as an environment variable
ENV ONNX_ROOT=/app/onnx

# Ensure the directory exists
RUN mkdir -p $ONNX_ROOT

# Run the FastAPI app with uvicorn
CMD ["python", "-m", "app.main"]