FROM python:3.11-slim

ARG GPU=false

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLOUDS_API_ENV=Production \
    FLOUDS_DEBUG_MODE=0

# Use consistent workdir name
WORKDIR /flouds-ai

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .

# Only install torch if GPU=true, otherwise rely on dependencies
RUN if [ "$GPU" = "true" ]; then \
      pip install --no-cache-dir torch; \
    fi

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Clean up system dependencies
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get purge -y build-essential && apt-get autoremove -y

# Copy application code
COPY app ./app

# Remove any embedded model files to keep image small
RUN find ./app -name "*.onnx" -type f -delete
RUN rm -rf /root/.cache/*

# Set port
EXPOSE 19690

# Create ONNX directory for mounting models
ENV FLOUDS_ONNX_ROOT=/flouds-ai/onnx
RUN mkdir -p $FLOUDS_ONNX_ROOT

# Create log directory 
ENV FLOUDS_LOG_PATH=/var/log/flouds
RUN mkdir -p $FLOUDS_LOG_PATH && \
    chmod 777 $FLOUDS_LOG_PATH

# Run the application
CMD ["python", "-m", "app.main"]