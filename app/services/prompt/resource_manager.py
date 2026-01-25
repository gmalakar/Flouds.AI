# =============================================================================
# File: resource_manager.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Model and session resource management for prompt service."""

import json
import os
from typing import Any, Optional, Set

import onnxruntime as ort

from app.app_init import APP_SETTINGS
from app.exceptions import ModelLoadError, ModelNotFoundError
from app.logger import get_logger
from app.services.prompt.models import MEMORY_LOW_THRESHOLD_MB, CachedSessions
from app.utils.log_sanitizer import sanitize_for_log

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
except Exception:
    ORTModelForSeq2SeqLM = None

logger = get_logger(__name__)


# ============================================================================
# Session Management
# ============================================================================


def load_special_tokens(special_tokens_path: str) -> Set[str]:
    """Load special tokens from JSON file.

    Args:
        special_tokens_path: Path to special tokens JSON file

    Returns:
        Set of special tokens, or empty set if file not found/invalid
    """
    logger.debug(
        "Loading special tokens from: %s", sanitize_for_log(special_tokens_path)
    )
    if not os.path.exists(special_tokens_path):
        logger.warning(f"Special tokens file not found: {special_tokens_path}")
        return set()

    try:
        from app.utils.path_validator import safe_open

        # Try to use safe_open if available, fallback to regular open
        try:
            root_path = os.path.dirname(special_tokens_path)
            with safe_open(special_tokens_path, root_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except TypeError:
            # safe_open not compatible, use standard open
            with open(special_tokens_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        tokens = set()
        for key in ["pad_token", "eos_token", "unk_token"]:
            if key in data and "content" in data[key]:
                tokens.add(data[key]["content"])

        if "additional_special_tokens" in data:
            if isinstance(data["additional_special_tokens"], list):
                tokens.update(data["additional_special_tokens"])
            elif isinstance(data["additional_special_tokens"], dict):
                tokens.update(data["additional_special_tokens"].values())

        return tokens
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in special tokens file: {e}")
        return set()
    except Exception as e:
        logger.error(f"Failed to load special tokens: {e}")
        return set()


def get_vocab_size_from_session(
    session: ort.InferenceSession,
) -> Optional[int]:
    """Extract vocabulary size from ONNX session output shape.

    Args:
        session: ONNX inference session

    Returns:
        Vocabulary size if detectable, None otherwise
    """
    try:
        import numpy as np

        outputs = session.get_outputs()
        if outputs:
            for output in outputs:
                # Look for logits output with shape [batch, seq_len, vocab_size]
                if output.shape and len(output.shape) >= 3:
                    vocab_dim = output.shape[-1]
                    if isinstance(vocab_dim, (int, np.integer)) and vocab_dim > 1000:
                        logger.debug(
                            f"Auto-detected vocab_size from ONNX output: {vocab_dim}"
                        )
                        return int(vocab_dim)
    except Exception as e:
        logger.warning(f"Could not auto-detect vocab_size from ONNX session: {e}")
    return None


def get_decoder_session(decoder_model_path: str) -> Optional[ort.InferenceSession]:
    """Get cached ONNX decoder session with error handling.

    Args:
        decoder_model_path: Path to decoder model

    Returns:
        ONNX inference session, or None if creation failed
    """
    try:
        provider = APP_SETTINGS.server.session_provider or "CPUExecutionProvider"
        available_providers = ort.get_available_providers()

        if provider not in available_providers:
            logger.warning(
                f"Provider {provider} not available for decoder, using CPUExecutionProvider"
            )
            provider = "CPUExecutionProvider"

        cache_key = f"{decoder_model_path}#{provider}"

        # First check if session is already cached
        cached_session = CachedSessions.decoder_sessions.get(cache_key)
        if cached_session is not None:
            return cached_session

        # Session not in cache - check memory and clear if needed
        from app.utils.cache_manager import CacheManager

        CacheManager.check_and_clear_cache_if_needed()

        logger.debug(
            "Creating decoder session for %s with provider %s",
            sanitize_for_log(decoder_model_path),
            sanitize_for_log(provider),
        )

        return CachedSessions.decoder_sessions.get_or_add(
            cache_key,
            lambda: ort.InferenceSession(decoder_model_path, providers=[provider]),
        )
    except Exception as e:
        logger.error(f"Failed to create decoder session: {e}")
        return None


def get_special_tokens(special_tokens_path: str) -> Set[str]:
    """Get cached special tokens.

    Args:
        special_tokens_path: Path to special tokens file

    Returns:
        Set of cached special tokens
    """
    return CachedSessions.special_tokens.get_or_add(
        special_tokens_path,
        lambda: load_special_tokens(special_tokens_path),
    )


def get_cached_model(model_path: str) -> Optional[Any]:
    """Get cached ONNX model with error handling.

    Args:
        model_path: Path to model

    Returns:
        Cached model instance, or None if loading failed

    Raises:
        ModelNotFoundError: If model file not found
        ModelLoadError: If model loading failed
    """
    if ORTModelForSeq2SeqLM is None:
        logger.warning("optimum[onnxruntime] not installed; skipping Seq2SeqLM load")
        return None

    try:
        # First check if model is already cached
        cached_model = CachedSessions.models.get(model_path)
        if cached_model is not None:
            return cached_model

        # Model not in cache - check memory and clear if needed
        from app.utils.cache_manager import CacheManager

        CacheManager.check_and_clear_cache_if_needed()

        logger.debug(f"Loading model from path: {model_path}")

        model = CachedSessions.models.get_or_add(
            model_path,
            lambda: ORTModelForSeq2SeqLM.from_pretrained(model_path, use_cache=False),
        )

        # Compatibility patch for newer transformers versions
        if not hasattr(model, "_supports_cache_class"):
            logger.debug("Applying compatibility patch for ORTModelForSeq2SeqLM")
            model._supports_cache_class = False

        logger.info(f"Model cache size: {CachedSessions.models.size()}")
        return model

    except FileNotFoundError:
        raise ModelNotFoundError(f"Model not found: {model_path}")
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}")


def clear_all_caches() -> None:
    """Clear all model/session/special token caches."""
    logger.info("Clearing all model/session/special token caches.")
    CachedSessions.clear_all()


def check_memory_and_clear_cache() -> None:
    """Check available memory and clear caches if below threshold."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        threshold_bytes = MEMORY_LOW_THRESHOLD_MB * 1024 * 1024

        if mem.available < threshold_bytes:
            logger.warning(
                f"Low memory before generation: {mem.available / 1024 / 1024:.1f}MB available"
            )
            from app.utils.cache_manager import CacheManager

            CacheManager.clear_all_caches()
    except Exception as e:
        logger.debug(f"Memory check failed: {e}")
