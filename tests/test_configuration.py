# =============================================================================
# File: test_configuration.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_configuration.py
# Date: copied from FloudsVector.Py tests/test_configuration.py
# =============================================================================

import os
import sys
from unittest.mock import Mock, patch  # noqa: F401

import pytest

# Add app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_app_settings():
    """Mock APP_SETTINGS for tests."""
    # Prefer updating the existing `app.app_init.APP_SETTINGS` shim (if present)
    # so modules that imported it earlier see the changes. Fall back to patch()
    # when the module isn't available.
    import sys
    import types

    app_init_mod = sys.modules.get("app.app_init")
    if app_init_mod and hasattr(app_init_mod, "APP_SETTINGS"):
        mock_settings = app_init_mod.APP_SETTINGS
    else:
        with patch("app.app_init.APP_SETTINGS") as mock_settings:
            pass

    # Configure default mock settings (ensure nested attributes exist)
    mock_settings.server = getattr(mock_settings, "server", types.SimpleNamespace())
    mock_settings.server.host = "localhost"
    mock_settings.server.port = 8080
    mock_settings.server.keepalive_timeout = getattr(mock_settings.server, "keepalive_timeout", 5)
    mock_settings.server.graceful_timeout = getattr(mock_settings.server, "graceful_timeout", 5)
    mock_settings.server.session_provider = getattr(
        mock_settings.server, "session_provider", "CPUExecutionProvider"
    )

    mock_settings.app = getattr(mock_settings, "app", types.SimpleNamespace())
    mock_settings.app.is_production = False
    mock_settings.app.debug = False
    mock_settings.app.max_request_size = getattr(mock_settings.app, "max_request_size", 26214400)
    mock_settings.app.request_timeout = getattr(mock_settings.app, "request_timeout", 30)
    mock_settings.app.cors_origins = getattr(mock_settings.app, "cors_origins", ["*"])

    mock_settings.security = getattr(mock_settings, "security", types.SimpleNamespace())
    mock_settings.security.enabled = getattr(mock_settings.security, "enabled", False)
    mock_settings.security.clients_db_path = getattr(
        mock_settings.security, "clients_db_path", None
    )
    mock_settings.security.trusted_hosts = getattr(mock_settings.security, "trusted_hosts", ["*"])
    mock_settings.security.enable_hsts = getattr(mock_settings.security, "enable_hsts", False)

    mock_settings.onnx = getattr(mock_settings, "onnx", types.SimpleNamespace())
    # Use a safe default path for ONNX models during tests
    import os

    default_onnx_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "onnx"))
    mock_settings.onnx.onnx_path = getattr(mock_settings.onnx, "onnx_path", default_onnx_root)

    mock_settings.vectordb = getattr(mock_settings, "vectordb", types.SimpleNamespace())
    mock_settings.vectordb.endpoint = getattr(mock_settings.vectordb, "endpoint", "localhost")
    mock_settings.vectordb.port = getattr(mock_settings.vectordb, "port", 19530)
    mock_settings.vectordb.username = getattr(mock_settings.vectordb, "username", "root")
    mock_settings.vectordb.password = getattr(mock_settings.vectordb, "password", "password")
    mock_settings.vectordb.default_dimension = getattr(
        mock_settings.vectordb, "default_dimension", 384
    )
    mock_settings.vectordb.admin_role_name = getattr(
        mock_settings.vectordb, "admin_role_name", "admin"
    )

    mock_settings.logging = getattr(mock_settings, "logging", types.SimpleNamespace())
    mock_settings.logging.folder = getattr(mock_settings.logging, "folder", "/tmp/logs")

    mock_settings.rate_limiting = getattr(mock_settings, "rate_limiting", types.SimpleNamespace())
    mock_settings.rate_limiting.enabled = getattr(mock_settings.rate_limiting, "enabled", False)
    mock_settings.rate_limiting.requests_per_minute = getattr(
        mock_settings.rate_limiting, "requests_per_minute", 0
    )
    mock_settings.rate_limiting.requests_per_hour = getattr(
        mock_settings.rate_limiting, "requests_per_hour", 0
    )

    yield mock_settings


def test_app_settings_shape(mock_app_settings):
    """Minimal test to assert the mocked `APP_SETTINGS` contains keys used by the app."""
    s = mock_app_settings
    # Common groups referenced by Flouds.Py
    assert hasattr(s, "server")
    assert hasattr(s, "app")
    assert hasattr(s, "security")
    # Security should expose these attributes (may be set elsewhere)
    assert hasattr(s.security, "enabled")
    assert hasattr(s.security, "clients_db_path")
    # CORS/trusted hosts defaults may be present
    # These are optional but middleware uses getattr with defaults
    # so presence is not strictly required; just ensure the object exists.
    assert hasattr(s, "logging")
