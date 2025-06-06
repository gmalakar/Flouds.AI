import json
import os

from app.config.appsettings import AppSettings
from app.config.onnx_config import OnnxConfig
from app.logger import get_logger

logger = get_logger("config_loader")


class ConfigLoader:
    _onnx_config_cache = None

    @staticmethod
    def get_app_settings() -> AppSettings:
        """
        Loads AppSettings from appsettings.json and environment-specific override in the same folder.
        Performs a deep merge for nested config sections.
        """
        data = ConfigLoader._load_config_data("appsettings.json", True)
        return AppSettings(**data)

    @staticmethod
    def get_onnx_config(key: str) -> OnnxConfig:
        """
        Loads OnnxConfig from onnx_config.json and environment-specific override in the same folder.
        Performs a deep merge for nested config sections.
        Only loads from file if config is not in cache.
        Returns the OnnxConfig for the specified key/model.
        Raises KeyError if the key is not found.
        """
        if ConfigLoader._onnx_config_cache is None:
            config_file_name = "onnx_config.json"
            data = ConfigLoader._load_config_data(config_file_name)
            ConfigLoader._onnx_config_cache = {
                k: OnnxConfig(**v) for k, v in data.items()
            }

        if key not in ConfigLoader._onnx_config_cache:
            raise KeyError(f"Model config '{key}' not found in onnx_config.json")
        return ConfigLoader._onnx_config_cache[key]

    @staticmethod
    def _load_config_data(config_file_name: str, check_env_file: bool = False) -> dict:
        """
        Loads a config file and merges with environment-specific override if present.
        Performs a deep merge for nested config sections.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(base_dir, config_file_name)

        logger.info(f"Loading config from {base_path}")

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Config file not found: {base_path}")

        with open(base_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Merge environment-specific config if requested and it exists (deep merge)
        if check_env_file:
            env = os.getenv("FASTAPI_ENV", "Production")
            name, ext = os.path.splitext(config_file_name)
            env_path = os.path.join(base_dir, f"{name}.{env.lower()}{ext}")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    env_data = json.load(f)
                deep_update(data, env_data)

        return data


# Example usage:
# settings = ConfigLoader.get_app_settings()
# onnx_cfg = ConfigLoader.get_onnx_config()
