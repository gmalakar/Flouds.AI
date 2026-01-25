# =============================================================================
# Multi-stage build for Flouds AI Service
# Builder stage: compile dependencies and clean up build artifacts
# Runtime stage: minimal runtime environment with only necessary files
# =============================================================================

# ==================== BUILDER STAGE ====================
FROM python:3.12-slim AS builder

ARG GPU=false
ARG TORCH_VERSION=2.9.1

# Build-time environment
ENV PIP_NO_CACHE_DIR=1 \
    PIP_NO_COMPILE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy requirements
COPY app/requirements-prod.txt /tmp/requirements-prod.txt

# Install dependencies in virtual environment
RUN set -ex \
    # Upgrade pip to fix CVE vulnerabilities
    && python -m pip install --upgrade --no-deps "pip>=25.3" \
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
    # Install all required dependencies
    && pip install --no-cache-dir -r /tmp/requirements-prod.txt \
    # Aggressive cleanup in builder to reduce copied size
    && find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type f -name '*.pyc' -delete \
    && find /opt/venv -type f -name '*.pyo' -delete \
    && find /opt/venv -type d -name test -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name tests -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name examples -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type f -name '*.a' -delete \
    && find /opt/venv -type f -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && rm -rf /tmp/* /root/.cache

# ==================== RUNTIME STAGE ====================
FROM python:3.12-slim

# ==================== RUNTIME STAGE ====================
FROM python:3.12-slim

# Runtime environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/flouds-ai \
    PATH=/opt/venv/bin:$PATH \
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
    FLOUDS_BG_CLEANUP_ENABLED=1 \
    FLOUDS_BG_CLEANUP_INTERVAL_SECONDS=60 \
    FLOUDS_BG_CLEANUP_INITIAL_JITTER_SECONDS=5 \
    FLOUDS_BG_CLEANUP_MAX_BACKOFF_SECONDS=600 \
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

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR ${PYTHONPATH}

# Copy application code
COPY app ./app

# Create runtime directories
RUN mkdir -p "$FLOUDS_ONNX_ROOT" "$FLOUDS_LOG_PATH" "$(dirname "$FLOUDS_CLIENTS_DB")"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /flouds-ai/app/healthcheck.py || exit 1

EXPOSE $FLOUDS_PORT

CMD ["python", "-m", "app.main"]
