import json

from app.config.appsettings import AppSettings, OnnxSettings
from app.config.config_loader import ConfigLoader


def test_appsettings_json_loads_into_model():
    """Ensure `appsettings.json` maps into `AppSettings` and nested `OnnxSettings`.

    This is a focused test that verifies the config loader reads the JSON
    file and Pydantic constructs the nested models with expected default
    values (as present in `app/config/appsettings.json`).
    """

    settings = ConfigLoader.get_app_settings()

    # Basic type checks
    assert isinstance(settings, AppSettings)
    assert isinstance(settings.onnx, OnnxSettings)

    # ONNX section values (from appsettings.json)
    assert isinstance(settings.onnx.config_file, str)
    assert settings.onnx.config_file.endswith("onnx_config.json")

    # Cache defaults from appsettings.json
    assert settings.cache.encoder_cache_max == 3
    assert settings.cache.decoder_cache_max == 3
    assert settings.cache.model_cache_max == 2
    assert settings.cache.special_tokens_cache_max == 8

    # Logging section should be present and contain expected keys
    log = settings.logging
    assert hasattr(log, "level")
    assert hasattr(log, "format")
