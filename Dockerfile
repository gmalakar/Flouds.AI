FROM python:3.11-slim

ARG GPU=false

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLOUDS_API_ENV=Production \
    FLOUDS_DEBUG_MODE=0

WORKDIR /flouds-ai

# Install system dependencies (OPTIONAL: Remove build-essential after pip install to reduce image size)
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .

# Install PyTorch for GPU or CPU
RUN if [ "$GPU" = "true" ]; then \
      pip install --no-cache-dir torch; \
    else \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install all other requirements (latest versions)
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for ONNX and other libraries
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# OPTIONAL: Remove build-essential after pip install to reduce image size
RUN apt-get purge -y build-essential && apt-get autoremove -y

COPY app ./app

RUN find ./app -name "*.onnx" -type f -delete
RUN rm -rf /root/.cache/*

EXPOSE 19690

ENV FLOUDS_ONNX_ROOT=/flouds-ai/onnx
RUN mkdir -p $FLOUDS_ONNX_ROOT

RUN find / -type f -size +10M -exec du -h {} + | sort -hr > /large_files.log || true

CMD ["python", "-m", "app.main"]