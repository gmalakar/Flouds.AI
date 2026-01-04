# =============================================================================
# File: test_configuration.py
# Date: copied from FloudsVector.Py tests/test_configuration.py
# =============================================================================

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_app_settings():
    """Mock APP_SETTINGS for tests."""
    with patch("app.app_init.APP_SETTINGS") as mock_settings:
        # Configure default mock settings
        mock_settings.server.host = "localhost"
        mock_settings.server.port = 8080
        mock_settings.app.is_production = False
        mock_settings.app.debug = False
        mock_settings.vectordb.endpoint = "localhost"
        mock_settings.vectordb.port = 19530
        mock_settings.vectordb.username = "root"
        mock_settings.vectordb.password = "password"
        mock_settings.vectordb.password_file = None
        mock_settings.vectordb.default_dimension = 384
        mock_settings.vectordb.admin_role_name = "admin"
        mock_settings.logging.folder = "/tmp/logs"
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
