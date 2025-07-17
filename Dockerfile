FROM python:3.11-slim

ARG GPU=false

ENV PYTHONUNBUFFERED=1 \
    FLOUDS_API_ENV=Production \
    FLOUDS_DEBUG_MODE=0 \
    FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
    FLOUDS_LOG_PATH=/flouds-ai/logs

WORKDIR /flouds-ai

# Copy requirements first for better layer caching
COPY app/requirements.txt .

# Install dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && if [ "$GPU" = "true" ]; then \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121; \
       else \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
       fi \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/* \
    && find /usr/local -name "*.pyc" -delete \
    && find /usr/local -name "__pycache__" -exec rm -rf {} + || true

# Copy application code
COPY app ./app

# Create user and directories
RUN mkdir -p $FLOUDS_ONNX_ROOT $FLOUDS_LOG_PATH \
    && chmod 777 $FLOUDS_LOG_PATH

# Add health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:19690/health || exit 1

EXPOSE 19690

CMD ["python", "-m", "app.main"]
