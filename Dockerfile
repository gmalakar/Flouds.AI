FROM python:3.12-slim

ARG GPU=false
ARG TORCH_VERSION=2.9.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/flouds-ai \
    # Basic environment
    FLOUDS_API_ENV=Production \
    APP_DEBUG_MODE=0 \
    # Networking - override at runtime if needed
    FLOUDS_HOST=0.0.0.0 \
    FLOUDS_PORT=19690 \
    # ONNX model root and optional ONNX config file (set in prod if required)
    FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
    FLOUDS_ONNX_CONFIG_FILE= \
    # Logging and persistence
    FLOUDS_LOG_PATH=/flouds-ai/logs \
    FLOUDS_CLIENTS_DB=/flouds-ai/data/clients.db \
    # Pip behavior
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_COMPILE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Cache sizing defaults (can be tuned via env)
    FLOUDS_ENCODER_CACHE_MAX=5 \
    FLOUDS_DECODER_CACHE_MAX=5 \
    FLOUDS_MODEL_CACHE_MAX=5 \
    FLOUDS_SPECIAL_TOKENS_CACHE_MAX=8 \
    FLOUDS_MODEL_CACHE_SIZE=10 \
    # Memory threshold (GB) for cache trimming
    FLOUDS_CACHE_MEMORY_THRESHOLD=1.0 \
    # Background cleanup defaults
    FLOUDS_BACKGROUND_CLEANUP_ENABLED=1 \
    FLOUDS_BACKGROUND_CLEANUP_INTERVAL_SECONDS=60 \
    FLOUDS_BACKGROUND_CLEANUP_INITIAL_JITTER_SECONDS=5 \
    FLOUDS_BACKGROUND_CLEANUP_MAX_BACKOFF_SECONDS=600 \
    FLOUDS_MEMORY_CHECK_INTERVAL=5 \
    # Request limits and timeouts
    FLOUDS_MAX_REQUEST_SIZE=10485760 \
    FLOUDS_REQUEST_TIMEOUT=30 \
    # Rate limiting (disabled by default)
    FLOUDS_RATE_LIMIT_ENABLED=0 \
    FLOUDS_RATE_LIMIT_PER_MINUTE=60 \
    FLOUDS_RATE_LIMIT_PER_HOUR=1000 \
    # Security / features
    FLOUDS_SECURITY_ENABLED=0 \
    FLOUDS_CONFIG_OVERRIDE=0 \
    # CORS / hosts (comma-separated lists; empty = no-op)
    FLOUDS_TRUSTED_HOSTS= \
    FLOUDS_CORS_ORIGINS= \
    # Logging rotation defaults
    FLOUDS_LOG_MAX_FILE_SIZE=10485760 \
    FLOUDS_LOG_BACKUP_COUNT=5 \
    FLOUDS_LOG_FORMAT="%(asctime)s %(levelname)s %(name)s: %(message)s"

# NOTE: Do NOT set secrets (e.g. FLOUDS_ENCRYPTION_KEY) in the Dockerfile.
# Provide secret values at runtime using your orchestration platform (K8s
# secrets, Docker secrets, or environment injection) so they are not baked
# into the image.

WORKDIR ${PYTHONPATH}

# Copy requirements first for better layer caching
COPY app/requirements-prod.txt requirements-prod.txt

# Install dependencies in single optimized layer
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && apt-get install -y --only-upgrade tar \
    # Upgrade pip to fix CVE vulnerabilities (pip <=25.2 affected)
    && python -m pip install --upgrade --no-cache-dir "pip>=25.3" \
    # Install PyTorch (CPU or GPU)
    && if [ "$GPU" = "true" ]; then \
        echo "Installing GPU PyTorch and ONNX Runtime GPU..."; \
        pip install --no-cache-dir --prefer-binary "torch==${TORCH_VERSION}" || true; \
        pip install --no-cache-dir --prefer-binary "onnxruntime-gpu>=1.18.0,<1.23.0"; \
    else \
        echo "Installing CPU PyTorch and ONNX Runtime CPU..."; \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            "torch==${TORCH_VERSION}+cpu" || true; \
        pip install --no-cache-dir --prefer-binary "onnxruntime>=1.18.0,<1.23.0"; \
    fi \
    # Install all required dependencies from requirements-prod.txt
    && pip install --no-cache-dir -r requirements-prod.txt \
    # Aggressive cleanup to minimize layer size
    && apt-get purge -y build-essential \
    && apt-get autoremove -y --purge \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /root/.cache \
        /tmp/* \
        /root/.pip-cache \
        /usr/share/doc/* \
        /usr/share/man/* \
        /usr/share/locale/* \
        /usr/share/info/* \
        /var/cache/apt/* \
        /var/log/* \
    # Remove Python compiled bytecode and cache directories
    && find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete \
    # Remove test directories and files
    && find /usr/local -type d -name test -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type d -name tests -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python*/site-packages -maxdepth 2 -type d -name tests -exec rm -rf {} + 2>/dev/null || true \
    # Remove examples and documentation directories
    && find /usr/local -type d -name examples -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python*/site-packages -maxdepth 2 -type d -name examples -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type d -name idle_test -exec rm -rf {} + 2>/dev/null || true \
    # Remove package metadata that's not needed at runtime
    && find /usr/local -type f \( -name '*.dist-info' -o -name '*.egg-info' \) -exec rm -rf {} + 2>/dev/null || true \
    # Remove static libraries and archives
    && find /usr/local -type f -name '*.a' -delete \
    # Remove pip cache and build artifacts
    && rm -rf /root/.cargo /root/.rustup \
    # Strip debugging symbols from shared libraries
    && find /usr/local -type f -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null || true \
    # Remove any remaining temporary build files
    && find /usr/local -type f -name '*.c' -delete 2>/dev/null || true \
    && find /usr/local -type f -name '*.h' ! -name 'pyconfig.h' -delete 2>/dev/null || true

# Copy application code
COPY app ./app

# Create runtime directories
RUN mkdir -p "$FLOUDS_ONNX_ROOT" "$FLOUDS_LOG_PATH" "$(dirname "$FLOUDS_CLIENTS_DB")"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /flouds-ai/app/healthcheck.py || exit 1

EXPOSE $FLOUDS_PORT

CMD ["python", "-m", "app.main"]
