# =============================================================================
# File: cache_keys.py
# Date: 2026-01-05
# =============================================================================

import os
from typing import Optional

from app.app_init import APP_SETTINGS
from app.config.config_loader import ConfigLoader
from app.utils.path_validator import validate_safe_path


def get_model_cache_key(model_to_use: str) -> str:
    """Return a canonical cache key for a model identifier.

    Behavior mirrors previous `BaseNLPService._get_model_cache_key`:
    - If a model config resolves to a local path (via `model_folder_name`),
      return the normalized absolute path.
    - If the provided identifier points to an existing filesystem path,
      return the normalized absolute path.
    - Otherwise return the identifier string stripped.

    This utility avoids importing `BaseNLPService` and is safe to call
    from other modules (prompt/embedder services) without creating
    circular imports.
    """
    try:
        resolved_path: Optional[str] = None
        cfg = None
        try:
            cfg = ConfigLoader.get_onnx_config(model_to_use)
        except Exception:
            cfg = None

        if cfg is not None:
            try:
                mf = getattr(cfg, "model_folder_name", None)
                if mf:
                    mf_norm = os.path.normpath(str(mf))
                    basename = os.path.basename(mf_norm)
                    model_variants = {
                        model_to_use,
                        str(model_to_use).replace("_", "-"),
                        str(model_to_use).replace("-", "_"),
                    }
                    if basename in model_variants:
                        rel_folder = mf_norm
                    else:
                        rel_folder = os.path.join(mf_norm, model_to_use)

                    root = str(getattr(APP_SETTINGS.onnx, "onnx_path", ""))
                    try:
                        full_path = validate_safe_path(
                            os.path.join(root, "models", rel_folder), root
                        )
                        resolved_path = full_path
                    except Exception:
                        resolved_path = None
            except Exception:
                resolved_path = None

        if not resolved_path and os.path.exists(model_to_use):
            resolved_path = model_to_use

        if resolved_path:
            try:
                return os.path.normpath(os.path.abspath(resolved_path))
            except Exception:
                return str(resolved_path)

        return str(model_to_use).strip()
    except Exception:
        return str(model_to_use).strip()


__all__ = ["get_model_cache_key"]
