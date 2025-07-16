FROM python:3.11-slim

ARG GPU=false

ENV PYTHONUNBUFFERED=1 \
    FLOUDS_API_ENV=Production \
    FLOUDS_DEBUG_MODE=0 \
    FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
    FLOUDS_LOG_PATH=/var/logs/flouds

WORKDIR /flouds-ai

COPY app/requirements.txt .

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

COPY app ./app

RUN adduser --disabled-password --gecos '' --uid 1000 flouds \
    && mkdir -p $FLOUDS_ONNX_ROOT $FLOUDS_LOG_PATH \
    && chown -R flouds:flouds /flouds-ai $FLOUDS_ONNX_ROOT $FLOUDS_LOG_PATH

USER flouds

EXPOSE 19690

CMD ["python", "-m", "app.main"]
