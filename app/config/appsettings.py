from typing import Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = Field(default="Flouds PY")


class ServerConfig(BaseModel):
    type: str = Field(default="uvicorn")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5001)
    reload: bool = Field(default=True)
    workers: int = Field(default=4)


class OnnxConfigSection(BaseModel):
    config_check_interval: int = Field(default=10)


class LoggingConfig(BaseModel):
    folder: str = Field(default="logs")
    app_log_file: str = Field(default="flouds.log")


class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    onnx: OnnxConfigSection = Field(default_factory=OnnxConfigSection)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
