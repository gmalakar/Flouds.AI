# =============================================================================
# File: export_logging.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

#!/usr/bin/env python3
"""Export logging helpers extracted from `export_model_consolidated.py`.

Provides setup/teardown functions that configure a per-export rotating
file logger and tee stdout/stderr to the logfile for easier debugging.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Tuple


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self.streams:
            try:
                if s and getattr(s, "isatty", lambda: False)():
                    return True
            except Exception:
                pass
        return False


def setup_export_logging(
    base_dir: str, safe_model: str, rev_tag: str, logger: logging.Logger
) -> Tuple[logging.Handler, object, object, object, Path]:
    """Create a rotating file handler and tee stdout/stderr to the log file.

    Returns: `(file_handler, logfile_fd, old_stdout, old_stderr, logfile_path)`
    Caller should call `teardown_export_logging` with the returned values.
    """
    logs_dir = Path(base_dir).parent / "logs" / "onnx_exports"
    logs_dir.mkdir(parents=True, exist_ok=True)

    import time

    ts = time.strftime("%Y%m%d-%H%M%S")
    logfile = logs_dir / f"{safe_model}_{rev_tag}_{ts}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        str(logfile), maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info("Logging to file: %s", logfile)

    # Also attach the file handler to the root logger so third-party
    # libraries that emit to other loggers (e.g. optimum/transformers)
    # are captured in the per-run logfile for easier debugging.
    try:
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    except Exception:
        pass

    logfile_fd = open(logfile, "a", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, logfile_fd)
    sys.stderr = Tee(sys.stderr, logfile_fd)

    return file_handler, logfile_fd, old_stdout, old_stderr, logfile


def teardown_export_logging(
    file_handler: logging.Handler,
    logfile_fd,
    old_stdout,
    old_stderr,
    logger: logging.Logger,
) -> None:
    """Restore stdout/stderr and remove the file handler."""
    try:
        if logfile_fd:
            try:
                logfile_fd.flush()
            except Exception:
                pass
            try:
                logfile_fd.close()
            except Exception:
                pass
        if old_stdout is not None:
            try:
                sys.stdout = old_stdout
            except Exception:
                pass
        if old_stderr is not None:
            try:
                sys.stderr = old_stderr
            except Exception:
                pass
    finally:
        try:
            if file_handler is not None:
                try:
                    logger.removeHandler(file_handler)
                except Exception:
                    pass
                try:
                    root_logger = logging.getLogger()
                    try:
                        root_logger.removeHandler(file_handler)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    file_handler.close()
                except Exception:
                    pass
        except Exception:
            pass
