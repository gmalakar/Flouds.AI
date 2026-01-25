# =============================================================================
# File: cache_registry.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: cache_registry.py
# Date: 2026-01-05
# =============================================================================

import os
import threading
from typing import Dict, Optional, cast

from app.modules.concurrent_dict import ConcurrentDict
from app.utils.simple_cache import SimpleCache

# Thread-local storage previously lived on `base_nlp_service`.
tokenizer_local = threading.local()

# Small model config cache (moved out of BaseNLPService so cache manager
# does not need to import the full service implementation).
MODEL_CONFIG_CACHE: SimpleCache = SimpleCache(max_size=50)

# The ConcurrentDict caches are initialized lazily to avoid import-order
# coupling with `APP_SETTINGS` and to allow tests to import this module
# without triggering full application initialization.
_INIT_LOCK = threading.Lock()
ENCODER_SESSIONS: Optional[ConcurrentDict] = None
DECODER_SESSIONS: Optional[ConcurrentDict] = None
MODELS: Optional[ConcurrentDict] = None
SPECIAL_TOKENS: Optional[ConcurrentDict] = None
GENERATION_CACHE: Optional[SimpleCache] = None
ENCODER_OUTPUT_CACHE: Optional[SimpleCache] = None
DECODE_CACHE: Optional[SimpleCache] = None
PROJECTION_MATRIX_CACHE: Optional[SimpleCache] = None


def _ensure_initialized() -> None:
    """Thread-safe lazy initializer for the shared concurrent caches.

    Reads configured limits from `APP_SETTINGS` when available, falling
    back to environment variables or sensible defaults when not.
    """
    global ENCODER_SESSIONS, DECODER_SESSIONS, MODELS, SPECIAL_TOKENS, GENERATION_CACHE, ENCODER_OUTPUT_CACHE, DECODE_CACHE, PROJECTION_MATRIX_CACHE

    if ENCODER_SESSIONS is not None:
        return

    with _INIT_LOCK:
        if ENCODER_SESSIONS is not None:
            return
        # Lazy import of APP_SETTINGS to avoid importing the whole application
        # during unit tests or when this module is imported for lightweight use.
        try:
            from app.app_init import APP_SETTINGS

            enc_limit = int(APP_SETTINGS.cache.encoder_cache_max)
            dec_limit = int(APP_SETTINGS.cache.decoder_cache_max)
            models_limit = int(APP_SETTINGS.cache.model_cache_max)
            special_limit = int(APP_SETTINGS.cache.special_tokens_cache_max)
        except Exception:
            enc_limit = int(os.getenv("FLOUDS_ENCODER_CACHE_MAX", "5"))
            dec_limit = int(os.getenv("FLOUDS_DECODER_CACHE_MAX", "5"))
            models_limit = int(os.getenv("FLOUDS_MODEL_CACHE_MAX", "5"))
            special_limit = int(os.getenv("FLOUDS_SPECIAL_TOKENS_CACHE_MAX", "8"))

        ENCODER_SESSIONS = ConcurrentDict("_encoder_sessions", max_size=enc_limit)
        DECODER_SESSIONS = ConcurrentDict("_decoder_sessions", max_size=dec_limit)
        MODELS = ConcurrentDict("_models", max_size=models_limit)
        SPECIAL_TOKENS = ConcurrentDict("_special_tokens", max_size=special_limit)
        # Generation cache: stores deterministic generation outputs keyed by
        # model/session id, encoder signature, input ids hash and generation params.
        try:
            gen_limit = int(APP_SETTINGS.cache.generation_cache_max)
        except Exception:
            gen_limit = 256
        GENERATION_CACHE = SimpleCache(max_size=gen_limit)
        try:
            enc_out_limit = int(APP_SETTINGS.cache.encoder_output_cache_max)
        except Exception:
            enc_out_limit = int(os.getenv("FLOUDS_ENCODER_OUTPUT_CACHE_MAX", "128"))
        ENCODER_OUTPUT_CACHE = SimpleCache(max_size=enc_out_limit)
        # Decode cache for decoded string results (tokenizer.decode outputs)
        try:
            decode_limit = int(APP_SETTINGS.cache.decode_cache_max)
        except Exception:
            decode_limit = int(os.getenv("FLOUDS_DECODE_CACHE_MAX", "1024"))
        DECODE_CACHE = SimpleCache(max_size=decode_limit)
        # Projection matrix cache: store small random matrices used for
        # embedding projection to ensure deterministic behavior and allow
        # clearing under memory pressure. Stored values are numpy arrays.
        try:
            proj_limit = int(APP_SETTINGS.cache.projection_matrix_cache_max)
        except Exception:
            proj_limit = int(os.getenv("FLOUDS_PROJECTION_CACHE_MAX", "128"))
        PROJECTION_MATRIX_CACHE = SimpleCache(max_size=proj_limit)


def ensure_initialized() -> None:
    """Public wrapper to lazily initialize caches (thread-safe).

    Exported for callers that want to guarantee caches are ready before
    performing direct attribute access. Prefer using the helper functions
    (`clear_*`, `get_cache_stats`) which perform initialization for you.
    """
    _ensure_initialized()


def clear_encoder_sessions() -> None:
    _ensure_initialized()
    cast(ConcurrentDict, ENCODER_SESSIONS).clear()


def clear_decoder_sessions() -> None:
    _ensure_initialized()
    cast(ConcurrentDict, DECODER_SESSIONS).clear()


def clear_models() -> None:
    _ensure_initialized()
    cast(ConcurrentDict, MODELS).clear()


def clear_special_tokens() -> None:
    _ensure_initialized()
    cast(ConcurrentDict, SPECIAL_TOKENS).clear()


def clear_generation_cache() -> None:
    _ensure_initialized()
    if GENERATION_CACHE is not None:
        GENERATION_CACHE.clear()


def clear_encoder_output_cache() -> None:
    _ensure_initialized()
    if ENCODER_OUTPUT_CACHE is not None:
        ENCODER_OUTPUT_CACHE.clear()


def clear_decode_cache() -> None:
    _ensure_initialized()
    if DECODE_CACHE is not None:
        DECODE_CACHE.clear()


def clear_projection_matrix_cache() -> None:
    _ensure_initialized()
    if PROJECTION_MATRIX_CACHE is not None:
        PROJECTION_MATRIX_CACHE.clear()


def clear_thread_tokenizers() -> None:
    if hasattr(tokenizer_local, "tokenizers"):
        tokenizer_local.tokenizers.clear()


def clear_model_config_cache() -> None:
    MODEL_CONFIG_CACHE.clear()


def get_encoder_sessions() -> ConcurrentDict:
    """Return the encoder sessions ConcurrentDict, initializing if needed."""
    _ensure_initialized()
    return cast(ConcurrentDict, ENCODER_SESSIONS)


def get_decoder_sessions() -> ConcurrentDict:
    """Return the decoder sessions ConcurrentDict, initializing if needed."""
    _ensure_initialized()
    return cast(ConcurrentDict, DECODER_SESSIONS)


def get_models_cache() -> ConcurrentDict:
    """Return the models ConcurrentDict, initializing if needed."""
    _ensure_initialized()
    return cast(ConcurrentDict, MODELS)


def get_special_tokens_cache() -> ConcurrentDict:
    """Return the special tokens ConcurrentDict, initializing if needed."""
    _ensure_initialized()
    return cast(ConcurrentDict, SPECIAL_TOKENS)


def get_generation_cache() -> SimpleCache:
    """Return shared generation SimpleCache, initializing if needed."""
    _ensure_initialized()
    return cast(SimpleCache, GENERATION_CACHE)


def get_encoder_output_cache() -> SimpleCache:
    """Return shared encoder output SimpleCache, initializing if needed."""
    _ensure_initialized()
    return cast(SimpleCache, ENCODER_OUTPUT_CACHE)


def get_decode_cache() -> SimpleCache:
    """Return shared decode SimpleCache, initializing if needed."""
    _ensure_initialized()
    return cast(SimpleCache, DECODE_CACHE)


def get_projection_matrix_cache() -> SimpleCache:
    """Return shared projection-matrix SimpleCache, initializing if needed."""
    _ensure_initialized()
    return cast(SimpleCache, PROJECTION_MATRIX_CACHE)


def get_cache_stats() -> Dict[str, int]:
    _ensure_initialized()
    return {
        "encoder_sessions": cast(ConcurrentDict, ENCODER_SESSIONS).size(),
        "decoder_sessions": cast(ConcurrentDict, DECODER_SESSIONS).size(),
        "models": cast(ConcurrentDict, MODELS).size(),
        "special_tokens": cast(ConcurrentDict, SPECIAL_TOKENS).size(),
        "model_configs": MODEL_CONFIG_CACHE.size(),
        "generation_cache": (GENERATION_CACHE.size() if GENERATION_CACHE is not None else 0),
        "encoder_output_cache": (
            ENCODER_OUTPUT_CACHE.size() if ENCODER_OUTPUT_CACHE is not None else 0
        ),
        "decode_cache": DECODE_CACHE.size() if DECODE_CACHE is not None else 0,
        "projection_matrix_cache": (
            PROJECTION_MATRIX_CACHE.size() if PROJECTION_MATRIX_CACHE is not None else 0
        ),
    }


def warm_up_model_configs(model_names: list[str]) -> None:
    """Pre-load model configurations into the shared MODEL_CONFIG_CACHE.

    This allows cache warm-up without importing `BaseNLPService`, avoiding
    circular imports between the service and cache manager.
    """
    try:
        from app.config.config_loader import ConfigLoader

        for name in model_names:
            try:
                cfg = ConfigLoader.get_onnx_config(name)
                if cfg is not None:
                    MODEL_CONFIG_CACHE.put(name, cfg)
            except Exception:
                # Do not raise here; warm-up is best-effort
                continue
    except Exception:
        # Best-effort; swallowing exceptions keeps warm-up non-fatal
        return


__all__ = [
    "tokenizer_local",
    "MODEL_CONFIG_CACHE",
    "ENCODER_SESSIONS",
    "DECODER_SESSIONS",
    "MODELS",
    "SPECIAL_TOKENS",
    "GENERATION_CACHE",
    "ENCODER_OUTPUT_CACHE",
    "DECODE_CACHE",
    "PROJECTION_MATRIX_CACHE",
    "clear_encoder_sessions",
    "clear_decoder_sessions",
    "clear_models",
    "clear_special_tokens",
    "clear_thread_tokenizers",
    "clear_model_config_cache",
    "clear_generation_cache",
    "clear_encoder_output_cache",
    "clear_decode_cache",
    "clear_projection_matrix_cache",
    "get_generation_cache",
    "get_encoder_output_cache",
    "get_decode_cache",
    "get_projection_matrix_cache",
    "get_cache_stats",
    "ensure_initialized",
]
