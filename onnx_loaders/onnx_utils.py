"""
Compatibility shim named `onnx_utils` to satisfy external optimizer imports.

This module exposes `extract_raw_data_from_model` (a minimal implementation
that returns an `onnx.ModelProto` when given a file path or model object) and
re-exports a few helper functions from the local `onnx_helpers.py`.

Why this exists: some third-party optimizer code imports `extract_raw_data_from_model`
from a top-level module named `onnx_utils`. The repo's internal helpers were
renamed to `onnx_helpers.py` to avoid collisions; this shim restores the
expected symbol names by delegating to the new helpers.
"""

from __future__ import annotations

import logging
from typing import Any

import onnx

# Import helpers with multiple fallbacks to work whether this file is loaded as
# a package module (`onnx_loaders.onnx_utils`) or as a top-level module
# (`onnx_utils`) depending on how sys.path is arranged at runtime.
try:
    # Preferred: package-relative import when running as part of the package
    from .onnx_helpers import (  # type: ignore
        create_ort_session,
        get_default_opset,
        get_logger,
        get_preferred_provider,
    )
except Exception:
    try:
        # When imported as a top-level module and the current directory is
        # `onnx_loaders`, this will succeed.
        from onnx_helpers import (  # type: ignore
            create_ort_session,
            get_default_opset,
            get_logger,
            get_preferred_provider,
        )
    except Exception:
        # Final fallback: try the full package path
        from onnx_loaders.onnx_helpers import (  # type: ignore
            create_ort_session,
            get_default_opset,
            get_logger,
            get_preferred_provider,
        )


_logger = get_logger(__name__)


def extract_raw_data_from_model(model_or_path: Any):
    """Return an ONNX ModelProto for the given input.

    This is a minimal implementation that satisfies import and basic usage
    patterns in downstream optimizer code. If `model_or_path` is a string it
    will be interpreted as a filepath and loaded via `onnx.load`. If it is
    already an ONNX `ModelProto`, it will be returned as-is.
    """
    try:
        if isinstance(model_or_path, str):
            _logger.debug("Loading ONNX model from path: %s", model_or_path)
            return onnx.load(model_or_path)
        # Assume it's already a ModelProto or similar object
        return model_or_path
    except Exception as e:
        _logger.exception("extract_raw_data_from_model failed: %s", e)
        raise


__all__ = [
    "extract_raw_data_from_model",
    "get_logger",
    "get_preferred_provider",
    "get_default_opset",
    "create_ort_session",
]


def has_external_data(model_or_path: Any) -> bool:
    """Return True if the given ONNX model (or filepath) references external data.

    This will first try to use `onnx.external_data_helper.has_external_data` when
    available, and fall back to a conservative manual inspection of initializers.
    """
    try:
        # Load model if a path was provided
        model = None
        if isinstance(model_or_path, str):
            import os

            if not os.path.exists(model_or_path):
                return False
            model = onnx.load(model_or_path)
        else:
            model = model_or_path

        # Prefer onnx helper if present
        try:
            from onnx import external_data_helper

            return external_data_helper.has_external_data(model)
        except Exception:
            pass

        # Fallback: inspect initializers for external data markers
        for init in getattr(model.graph, "initializer", []):
            # TensorProto.data_location == TensorProto.EXTERNAL indicates external data
            if getattr(init, "data_location", None) == getattr(
                onnx.TensorProto, "EXTERNAL", 1
            ):
                return True
            # external_data repeated field present
            if getattr(init, "external_data", None):
                return True

        return False
    except Exception:
        _logger.exception("has_external_data check failed")
        return False


__all__.append("has_external_data")
# The helper functions `get_logger`, `get_preferred_provider`, `get_default_opset`,
# and `create_ort_session` are imported above from `onnx_helpers` to avoid duplication.
