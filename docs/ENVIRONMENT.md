````markdown
# Environment variables

This file documents environment variables used by the `Flouds.Py` project, their defaults, whether they are required, and where they are referenced. Use this to build your `.env` file and Dockerfile.

Format: `NAME` — Default — Required? — Where used — Short description

- `FLOUDS_API_ENV` — `Production` — Optional (recommended set in deployments) — `app/app_init.py`, `app/config/config_loader.py`, `app/logger.py` — Runtime environment: `Development`/`Production`/`Enterprise`. Affects validation and logging behavior.
- `APP_DEBUG_MODE` — `0` — Optional — `app/main.py`, `app/logger.py` — Toggle debug mode (`1` enables debug logging/features).
- `FLOUDS_LOG_PATH` — no default (implementation default used) — Optional — `app/logger.py`, `app/config/config_loader.py`, `app/app_init.py` — Directory for application logs.
- `FLOUDS_LOG_MAX_FILE_SIZE` — `10485760` — Optional — `app/logger.py` — Max bytes for rotating log file.
- `FLOUDS_LOG_BACKUP_COUNT` — `5` — Optional — `app/logger.py` — Number of rotated log files to keep.
- `FLOUDS_CLIENTS_DB` — `clients.db` — Optional (recommended set) — `app/app_init.py`, `app/config/config_loader.py`, `app/key_manager.py` — Path/name of the clients DB file.
- `FLOUDS_ONNX_ROOT` — none — Optional — `app/app_init.py`, `app/config/config_loader.py` — Root folder for ONNX models. Optional in Development; required in Production if ONNX usage expected.
- `FLOUDS_ONNX_CONFIG_FILE` — none — Optional — `app/app_init.py` — ONNX config filename (optional in Development).
- `ONNX_PATH` — `../onnx` (used by export utilities) — Optional — `onnx_loaders/export_model.py`, `onnx_loaders/export_model_v2.py` — Fallback path for ONNX export scripts.
- `FLOUDS_ENCRYPTION_KEY` — none — Required in Production for secure operations — `app/services/config_service.py`, `app/modules/key_manager.py` — Encryption key for tokens/secrets. Treat as sensitive (store in vault/CI secrets).
- `FLOUDS_ENCODER_CACHE_MAX` — `3` — Optional — `app/services/base_nlp_service.py`, `app/services/cache_registry.py` — Max entries in encoder session cache.
- `FLOUDS_DECODER_CACHE_MAX` — `3` — Optional — `app/services/prompt_service.py`, `app/services/cache_registry.py` — Max entries in decoder/session cache.
- `FLOUDS_MODEL_CACHE_MAX` — `2` — Optional — `app/services/prompt_service.py`, `app/services/cache_registry.py` — Max entries in model metadata cache.
- `FLOUDS_SPECIAL_TOKENS_CACHE_MAX` — `8` — Optional — `app/services/prompt_service.py`, `app/services/cache_registry.py` — Max entries in special-tokens cache.
- `FLOUDS_MEMORY_CHECK_INTERVAL` — `5` — Optional — `app/utils/cache_manager.py` — Interval in seconds for periodic memory checks in cache monitor.
- `FLOUDS_CACHE_MEMORY_THRESHOLD` — `1.0` — Optional — `app/utils/cache_manager.py` — Memory threshold (GB) used by cache cleanup logic.
- `FLOUDS_CORS_ORIGINS` — none — Optional — `app/app_startup.py` — Comma-separated list of allowed CORS origins.
- `FLOUDS_TRUSTED_HOSTS` — none — Optional — `app/app_startup.py` — Comma-separated list of trusted hosts.
- `FLOUDS_CONFIG_OVERRIDE` — `0` — Optional — `app/app_startup.py` — If `1`, allow config overrides at startup.
- `HEALTHCHECK_URL` — none — Optional — `app/healthcheck.py` — Explicit healthcheck URL override.
- `HEALTHCHECK_HOST` — falls back to `SERVER_HOST` or `localhost` — Optional — `app/healthcheck.py` — Healthcheck host.
- `HEALTHCHECK_PORT` — falls back to `SERVER_PORT` or `19690` — Optional — `app/healthcheck.py` — Healthcheck port.
- `HEALTHCHECK_PATH` — `/api/v1/health` — Optional — `app/healthcheck.py` — Health endpoint path.
- `HEALTHCHECK_TIMEOUT` — `8` — Optional — `app/healthcheck.py` — Healthcheck timeout in seconds.
- `SERVER_HOST` — none — Optional — fallback used by healthcheck & config
- `SERVER_PORT` — none — Optional — fallback used by healthcheck & config

Notes and recommendations

- Secrets: `FLOUDS_ENCRYPTION_KEY` (and any other keys) must be provided securely (CI secrets manager, Docker secrets, or environment injection). Do not commit secrets to source control.
- Defaults: Many vars have safe defaults for local Development; set `FLOUDS_API_ENV=Development` when running tests or in local dev to relax production-only validations.
- Logging: `APP_DEBUG_MODE=1` enables debugging output. For CI and Production, prefer `APP_DEBUG_MODE=0`.

Example `.env` (development)
```env
# Runtime
FLOUDS_API_ENV=Development
APP_DEBUG_MODE=1

# Logging
FLOUDS_LOG_PATH=./logs
FLOUDS_LOG_MAX_FILE_SIZE=10485760
FLOUDS_LOG_BACKUP_COUNT=5

# Caching
FLOUDS_ENCODER_CACHE_MAX=3
FLOUDS_DECODER_CACHE_MAX=3
FLOUDS_MODEL_CACHE_MAX=2
FLOUDS_SPECIAL_TOKENS_CACHE_MAX=8

# ONNX (optional in Development)
FLOUDS_ONNX_ROOT=
FLOUDS_ONNX_CONFIG_FILE=

# Secrets (do not commit)
FLOUDS_ENCRYPTION_KEY=
```

Example Dockerfile snippet
```dockerfile
ENV FLOUDS_API_ENV=Production
ENV APP_DEBUG_MODE=0
ENV FLOUDS_LOG_PATH=/var/log/flouds
# Inject secrets via build-time args or Docker secrets in production
```

If you'd like, I can add this file to the repo (create `ENVIRONMENT.md`) and/or generate a `.env.example` and Dockerfile snippets committed to the repo. Tell me which files you want added.

````
