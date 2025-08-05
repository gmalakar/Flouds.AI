# =============================================================================
# File: appsettings.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
import os
from typing import List, Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = Field(default="Flouds AI")
    version: str = Field(default="1.0.0")
    description: str = Field(
        default="AI-powered text summarization and embedding service"
    )
    debug: bool = Field(default=False)
    working_dir: str = Field(default=os.getcwd())
    is_production: bool = Field(default=True)
    cors_origins: List[str] = Field(default=["*"])
    max_request_size: int = Field(default=10485760)  # 10MB
    request_timeout: int = Field(default=300)  # 5 minutes


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=19690)
    session_provider: str = Field(default="CPUExecutionProvider")
    keepalive_timeout: int = Field(default=5)
    graceful_timeout: int = Field(default=30)


class OnnxConfigSection(BaseModel):
    onnx_path: Optional[str] = None
    config_file: str = Field(default="onnx_config.json")
    model_cache_size: int = Field(default=5)
    model_cache_ttl: int = Field(default=3600)
    enable_optimizations: bool = Field(default=True)


class RateLimitConfig(BaseModel):
    enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=200)
    requests_per_hour: int = Field(default=5000)
    cleanup_interval: int = Field(default=300)


class MonitoringConfig(BaseModel):
    enable_metrics: bool = Field(default=True)
    memory_threshold_mb: int = Field(default=1024)
    cpu_threshold_percent: int = Field(default=80)


class SecurityConfig(BaseModel):
    enabled: bool = Field(default=False)
    clients_db_path: str = Field(default="clients.db")


class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    onnx: OnnxConfigSection = Field(default_factory=OnnxConfigSection)

    rate_limiting: RateLimitConfig = Field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
