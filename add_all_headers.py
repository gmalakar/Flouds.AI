# =============================================================================
# File: add_all_headers.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import glob
import subprocess

py_files = glob.glob("**/*.py", recursive=True)
for f in py_files:
    if ".venv" not in f and "__pycache__" not in f:
        subprocess.run(["python", "add_header.py", f])
