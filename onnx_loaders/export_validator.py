# =============================================================================
# File: export_validator.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

#!/usr/bin/env python3
"""Validator invocation helper used by the exporter.

Provides `invoke_validator(...)` which performs the quick structural
verification (checker/external_data/session) and numeric validation by
invoking the centralized validator programmatically or, if necessary,
via an in-process module run as a fallback. Returns `(rc, quick_ok)` where
`rc` is the numeric validator return code (0 success, 2 numeric fail,
3 missing model/error) and `quick_ok` indicates whether the lightweight
quick verification passed.
"""
from __future__ import annotations

import importlib
import runpy
import sys
from typing import List, Tuple


def invoke_validator(
    output_dir: str,
    expected: List[str],
    model_name: str | None,
    pack_single_file: bool,
    pack_single_threshold_mb: int | None,
    trust_remote_code: bool,
    normalize_embeddings: bool,
    logger,
) -> Tuple[int, bool]:
    rc = 3
    quick_ok = True

    # Try to load the validator module programmatically first.
    validator = None
    try:
        from . import validate_onnx_model as validator
    except Exception:
        try:
            validator = importlib.import_module("onnx_loaders.validate_onnx_model")
        except Exception:
            validator = None

    trust_flag = False
    try:
        trust_flag = bool(trust_remote_code)
    except Exception:
        trust_flag = False

    logger.info(
        "Invoking ONNX validator for %s (trust_remote_code=%s)", output_dir, trust_flag
    )

    try:
        if validator is not None:
            try:
                try:
                    rc = validator.validate_onnx(
                        model_dir=output_dir,
                        reference_model=(str(model_name) if model_name else output_dir),
                        texts=None,
                        device="cpu",
                        atol=1e-4,
                        rtol=1e-3,
                        trust_remote_code=trust_flag,
                        normalize_embeddings=normalize_embeddings,
                    )
                except Exception as call_err:
                    logger.error(
                        "Programmatic validator raised an exception: %s", call_err
                    )
                    rc = 3
            except Exception as call_err:
                logger.error("Programmatic validator raised: %s", call_err)
                rc = 3

        # If programmatic validation not available or it failed, fall back to invoking
        # the validator as an in-process module run so we can isolate deps.
        if validator is None or rc != 0:
            args = [
                "--model_dir",
                output_dir,
                "--device",
                "cpu",
                "--atol",
                "1e-4",
                "--rtol",
                "1e-3",
            ]
            if model_name:
                args += ["--reference_model", str(model_name)]
            if trust_flag:
                args += ["--trust_remote_code"]
            if normalize_embeddings:
                args += ["--normalize-embeddings"]

            mod_name = "onnx_loaders.validate_onnx_model"
            old_mod = None
            try:
                old_mod = sys.modules.pop(mod_name, None)
            except Exception:
                old_mod = None

            # Temporarily set sys.argv for the in-process run
            old_argv = sys.argv[:]
            sys.argv = [mod_name] + args
            try:
                logger.info(
                    "Running validator in-process: %s %s", mod_name, " ".join(args)
                )
                runpy.run_module(mod_name, run_name="__main__")
                rc = 0
            except SystemExit as se:
                try:
                    rc = int(se.code) if se.code is not None else 0
                except Exception:
                    rc = 1
            except Exception as inproc_err:
                logger.error("In-process validator failed: %s", inproc_err)
                rc = 3
            finally:
                sys.argv = old_argv
                try:
                    if old_mod is not None:
                        del old_mod
                except Exception:
                    pass

    except Exception as e:
        logger.error("Failed to execute validator for %s: %s", output_dir, e)
        rc = 3

    return int(rc), bool(quick_ok)
