"""Test package initializer that registers `tests.test_configuration` as
pytest plugin so fixtures defined there are available without creating
a `test_configuration.py` file.

Ensure the repository root is on `sys.path` so pytest can import
`tests.test_configuration` reliably when running single-file tests.
"""

import os
import sys

# Add repository root (parent of the `tests` package) to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest_plugins = ["tests.test_configuration"]
