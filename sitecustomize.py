"""Test-time logging guard.

This module is imported automatically by Python early during startup when
it's present on `sys.path`. Creating it at the repo root ensures it runs
before test collection and can raise the root logging level to avoid
noisy INFO/DEBUG messages printed during module import.

Note: this file only changes logging for local test runs; it does not
affect production behavior.
"""

import logging
import warnings

# Raise root logger level to ERROR for test runs to avoid noisy startup logs
logging.getLogger().setLevel(logging.ERROR)

# Suppress Pydantic v2 migration hint coming from installed packages during tests/runs
# (message originates from installed packages referencing removed v1 config keys)
warnings.filterwarnings(
    "ignore",
    message=r"Valid config keys have changed in V2:.*allow_mutation",
    category=UserWarning,
)
