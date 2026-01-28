# =============================================================================
# File: test_config_loader_csp.py
# Date: 2026-01-27
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

import json

from app.config.config_loader import ConfigLoader


def test_csp_env_override_comma(monkeypatch):
    # Ensure we don't require production ONNX files during test
    monkeypatch.setenv("FLOUDS_API_ENV", "Development")

    # Avoid reading actual appsettings.json on disk; return empty config
    monkeypatch.setattr(ConfigLoader, "_load_config_data", lambda *a, **k: {})

    # Clear any cached settings
    ConfigLoader.clear_cache()

    # Provide a comma-separated override
    monkeypatch.setenv("FLOUDS_SECURITY_CSP_SCRIPT_SRC", "'self',https://cdn.jsdelivr.net")

    settings = ConfigLoader.get_app_settings()

    assert settings.security.csp_script_src == ["'self'", "https://cdn.jsdelivr.net"]


def test_csp_env_override_json(monkeypatch):
    monkeypatch.setenv("FLOUDS_API_ENV", "Development")
    monkeypatch.setattr(ConfigLoader, "_load_config_data", lambda *a, **k: {})
    ConfigLoader.clear_cache()

    # Provide a JSON array override
    val = json.dumps(["'self'", "https://cdn.jsdelivr.net"])
    monkeypatch.setenv("FLOUDS_SECURITY_CSP_STYLE_SRC", val)

    settings = ConfigLoader.get_app_settings()

    assert settings.security.csp_style_src == ["'self'", "https://cdn.jsdelivr.net"]
