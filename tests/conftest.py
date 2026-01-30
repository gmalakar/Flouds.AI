# =============================================================================
# File: conftest.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import asyncio
import atexit
import logging
import os
import sys
import types
import warnings

# Provide a lightweight `app.app_init` shim at import time to avoid running
# ConfigLoader during test collection. Individual tests may patch this object
# further via fixtures as needed.
if "app.app_init" not in sys.modules:
    shim = types.ModuleType("app.app_init")
    shim.APP_SETTINGS = types.SimpleNamespace(
        server=types.SimpleNamespace(
            host="localhost",
            port=8080,
            keepalive_timeout=5,
            graceful_timeout=5,
            session_provider="CPUExecutionProvider",
        ),
        app=types.SimpleNamespace(
            name="Flouds Test",
            description="Test",
            version="0.0.0",
            is_production=False,
            debug=True,
            max_request_size=26214400,
            request_timeout=30,
            cors_origins=["*"],
        ),
        security=types.SimpleNamespace(
            enabled=False, clients_db_path=None, trusted_hosts=["*"], enable_hsts=False
        ),
        onnx=types.SimpleNamespace(onnx_path=os.getcwd()),
        vectordb=types.SimpleNamespace(
            endpoint="localhost",
            port=19530,
            username="root",
            password="password",
            default_dimension=384,
            admin_role_name="admin",
        ),
        logging=types.SimpleNamespace(folder="/tmp/logs"),
        rate_limiting=types.SimpleNamespace(
            enabled=False, requests_per_minute=0, requests_per_hour=0
        ),
    )
    sys.modules["app.app_init"] = shim

import pytest

from tests.test_configuration import mock_app_settings  # noqa: F401

# Prefer selector policy in tests to avoid ProactorEventLoop lifecycle warnings on Windows
if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Suppress noisy event loop ResourceWarnings emitted at interpreter shutdown
warnings.filterwarnings("ignore", message="unclosed event loop", category=ResourceWarning)


def _close_loop_if_open(loop: asyncio.AbstractEventLoop) -> None:
    if loop is None or loop.is_closed():
        return
    try:
        if not loop.is_running():
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
        loop.close()
    except Exception:
        pass


def _close_known_event_loops(reset_default: bool = False) -> None:
    policy = asyncio.get_event_loop_policy()
    local = getattr(policy, "_local", None)
    loops = set()
    if local is not None:
        for value in getattr(local, "__dict__", {}).values():
            if isinstance(value, asyncio.AbstractEventLoop):
                loops.add(value)
        loops.update({getattr(local, "loop", None), getattr(local, "_loop", None)})
    loops.add(getattr(policy, "_loop", None))
    try:
        loops.add(policy.get_event_loop())
    except Exception:
        pass
    for candidate in loops:
        _close_loop_if_open(candidate)
    if reset_default:
        try:
            fresh = asyncio.new_event_loop()
            asyncio.set_event_loop(fresh)
            _close_loop_if_open(fresh)
        except Exception:
            pass


atexit.register(lambda: _close_known_event_loops(reset_default=True))


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


@pytest.fixture(autouse=True, scope="session")
def _close_event_loop_at_session_end():
    """Ensure any lingering event loop created during tests is closed."""
    yield
    _close_known_event_loops()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    # Final safety net to close any remaining event loops
    _close_known_event_loops(reset_default=True)


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
