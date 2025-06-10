# =============================================================================
# File: appsettings.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = Field(default="Flouds PY")
    debug: bool = Field(default=False)


class ServerConfig(BaseModel):
    type: str = Field(default="uvicorn")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5001)
    reload: bool = Field(default=True)
    workers: int = Field(default=4)
    model_session_provider: str = Field(default="CPUExecutionProvider")


class OnnxConfigSection(BaseModel):
    rootpath: str = Field(default="./app/onnx")


class LoggingConfig(BaseModel):
    folder: str = Field(default="logs")
    app_log_file: str = Field(default="flouds.log")


class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    onnx: OnnxConfigSection = Field(default_factory=OnnxConfigSection)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
