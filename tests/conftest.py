# =============================================================================
# File: conftest.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import logging

import pytest

from tests.test_configuration import mock_app_settings  # noqa: F401


@pytest.fixture(autouse=True, scope="session")
def silence_noisy_loggers():
    """Raise log level for noisy module loggers during tests.

    Some modules (e.g., auth/config loader) emit expected warnings/info
    during test runs which clutter output. This fixture sets their
    logger level to ERROR for the duration of the test session.
    """
    noisy_loggers = [
        "flouds.auth",
        "flouds.app_init",
        "flouds.config_loader",
    ]
    previous_levels = {}
    for name in noisy_loggers:
        logger = logging.getLogger(name)
        previous_levels[name] = logger.level
        logger.setLevel(logging.ERROR)

    yield

    # restore previous levels
    for name, level in previous_levels.items():
        logging.getLogger(name).setLevel(level)


def pytest_configure(config):
    """Ensure noisy loggers are at ERROR level after pytest config loads.

    Some test plugins or modules may reconfigure logging during collection;
    setting levels here ensures our suppression takes effect for the whole
    session.
    """
    noisy_loggers = [
        "flouds.auth",
        "flouds.app_init",
        "flouds.config_loader",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)


# Minimal conftest to expose fixtures for full-suite pytest runs.
# It re-exports `mock_app_settings` defined in `tests/test_configuration.py`.


@pytest.fixture(autouse=True)
def _patch_onnxruntime(monkeypatch):
    """Patch onnxruntime.InferenceSession with a lightweight fake during tests.

    Many unit tests only need a session-like object that supports `.run()`
    and `.get_outputs()` and do not require loading a real ONNX file. This
    fixture prevents tests from failing when model files are not present.
    """
    try:
        import onnxruntime as _ort
    except Exception:
        _ort = None

    import numpy as _np

    class _FakeSession:
        def __init__(self, model_path, *args, **kwargs):
            self.model_path = model_path

        def run(self, *args, **kwargs):
            # Return a plausible encoder output shape by default
            return [_np.zeros((1, 8, 512))]

        def get_outputs(self):
            class _Out:
                def __init__(self, name):
                    self.name = name

            return [_Out("output")]

    if _ort is not None:
        monkeypatch.setattr(_ort, "InferenceSession", _FakeSession)

    yield
