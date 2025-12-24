FROM python:3.12-slim

ARG GPU=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/flouds-ai \
    FLOUDS_API_ENV=Production \
    APP_DEBUG_MODE=0 \
    FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
    FLOUDS_LOG_PATH=/flouds-ai/logs \
    FLOUDS_CLIENTS_DB=/flouds-ai/tinydb/clients.db \
    PIP_NO_CACHE_DIR=1 \
    FLOUDS_ENCODER_CACHE_MAX=3 \
    FLOUDS_DECODER_CACHE_MAX=3 \
    FLOUDS_MODEL_CACHE_MAX=2 \
    FLOUDS_SPECIAL_TOKENS_CACHE_MAX=8 \
    FLOUDS_CACHE_MEMORY_THRESHOLD=1.0

# Upgrade pip to fix CVE vulnerabilities (pip <=25.2 affected)
RUN python -m pip install --upgrade "pip>=25.3"
WORKDIR ${PYTHONPATH}

# Copy requirements first for better layer caching
COPY app/requirements.txt .

# Install dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get install -y --only-upgrade tar \
    && if [ "$GPU" = "true" ]; then \
    echo "Installing ONNX Runtime GPU..."; \
    pip install --no-cache-dir --prefer-binary onnxruntime-gpu==1.17.3; \
    else \
    echo "Installing ONNX Runtime CPU..."; \
    pip install --no-cache-dir --prefer-binary onnxruntime==1.17.3; \
    fi \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/* /root/.pip-cache \
    && find /usr/local -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type d -name "test" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name "*.pyc" -delete \
    && find /usr/local -type f -name "*.pyo" -delete \
    && find /usr/local -type f \( -name "*.dist-info" -o -name "*.egg-info" \) -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python*/site-packages -maxdepth 2 -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python*/site-packages -maxdepth 2 -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY app ./app

# Create runtime directories
RUN mkdir -p "$FLOUDS_ONNX_ROOT" "$FLOUDS_LOG_PATH" "$(dirname "$FLOUDS_CLIENTS_DB")"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /flouds-ai/app/healthcheck.py || exit 1

EXPOSE 19690

CMD ["python", "-m", "app.main"]
