# =============================================================================
# File: appsettings.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
from typing import List, Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = Field(default="Flouds AI")
    version: str = Field(default="1.0.0")
    description: str = Field(default="AI-powered text summarization and embedding service")
    debug: bool = Field(default=False)
    is_production: bool = Field(default=True)
    cors_origins: List[str] = Field(default=["*"])
    max_request_size: int = Field(default=10485760)  # 10MB
    request_timeout: int = Field(default=300)  # 5 minutes


class ServerConfig(BaseModel):
    # Default to localhost to avoid unintentionally binding to all interfaces
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=19690)
    session_provider: str = Field(default="CPUExecutionProvider")
    keepalive_timeout: int = Field(default=5)
    graceful_timeout: int = Field(default=30)


class OnnxSettings(BaseModel):
    onnx_path: Optional[str] = None
    config_file: str = Field(default="onnx_config.json")


class RateLimitConfig(BaseModel):
    enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=200)
    requests_per_hour: int = Field(default=5000)
    cleanup_interval: int = Field(default=300)


class MonitoringConfig(BaseModel):
    enable_metrics: bool = Field(default=True)
    memory_threshold_mb: int = Field(default=1024)
    cpu_threshold_percent: int = Field(default=80)
    cache_cleanup_max_age_seconds: int = Field(default=60)
    # Background cleanup monitor settings
    enable_background_cleanup: bool = Field(
        default=False,
        description="Enable proactive background cache cleanup worker",
    )
    background_cleanup_interval_seconds: int = Field(
        default=60,
        description="Default interval in seconds between cleanup runs",
    )
    background_cleanup_initial_jitter_seconds: int = Field(
        default=5,
        description="Maximum random jitter applied to initial start delay",
    )
    background_cleanup_max_backoff_seconds: int = Field(
        default=300,
        description="Maximum backoff in seconds when repeated errors occur",
    )


class SecurityConfig(BaseModel):
    """
    Security configuration settings.
    """

    enabled: bool = Field(default=False)
    clients_db_path: str = Field(default="/app/data/clients.db")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="List of allowed CORS origins. Use '*' to allow all.",
    )
    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["*"],
        description="List of trusted hostnames for TrustedHostMiddleware. Use '*' to allow all.",
    )


class LoggingConfig(BaseModel):
    """
    Logging configuration settings. This mirrors the `logging` section
    found in `appsettings.json` so those values are validated and
    populated into `AppSettings`.
    """

    level: str = Field(default="INFO", description="Logging level")
    max_file_size: int = Field(default=10485760, description="Maximum size in bytes for log files")
    backup_count: int = Field(default=5, description="Number of backup files")
    format: str = Field(
        default="%(asctime)s %(levelname)s %(name)s: %(message)s",
        description="Log record format",
    )


class CacheConfig(BaseModel):
    """
    Runtime cache limits for in-memory caches.

    These values are intended to be small integers controlling the maximum
    number of entries retained in each shared cache. They can be overridden
    by environment variables at startup via `ConfigLoader`.
    """

    encoder_cache_max: int = Field(
        default=3,
        description="Maximum number of encoder ONNX sessions to cache",
    )
    decoder_cache_max: int = Field(
        default=3,
        description="Maximum number of decoder ONNX sessions to cache",
    )
    model_cache_max: int = Field(
        default=2,
        description="Maximum number of Optimum/seq2seq models to cache",
    )
    special_tokens_cache_max: int = Field(
        default=8,
        description="Maximum number of special tokens map entries to cache",
    )
    generation_cache_max: int = Field(
        default=256,
        description="Maximum number of deterministic generation outputs to cache",
    )
    encoder_output_cache_max: int = Field(
        default=128,
        description="Maximum number of encoder outputs to cache (seq2seq)",
    )
    encoder_output_cache_max_array_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum byte-size of a single encoder output array to cache",
    )
    decode_cache_max: int = Field(
        default=1024,
        description="Maximum number of decode output arrays to cache",
    )
    projection_matrix_cache_max: int = Field(
        default=128,
        description="Maximum number of projection matrix arrays to cache",
    )


class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    onnx: OnnxSettings = Field(default_factory=OnnxSettings)

    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    rate_limiting: RateLimitConfig = Field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
