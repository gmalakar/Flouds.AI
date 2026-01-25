# =============================================================================
# File: resource_manager.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Resource management for tokenizers, ONNX sessions, and model configs."""

import copy
import os
from types import SimpleNamespace
from typing import Any, Tuple

from app.exceptions import ModelLoadError, ModelNotFoundError, TokenizerError
from app.logger import get_logger
from app.services.base_nlp_service import BaseNLPService
from app.services.embedder.onnx_utils import (
    get_native_dimension_from_session,
    get_output_names_from_session,
)

# Import for use later - allows test mocking via app.services.embedder_service
from app.utils.path_validator import validate_safe_path

logger = get_logger("embedder_service.resources")


def prepare_embedding_resources(
    model: str, base_service_class: Any
) -> Tuple[Any, Any, Any]:
    """Prepare model config, tokenizer, and session for embedding.

    Args:
        model: Model name/identifier
        base_service_class: BaseNLPService class reference (for static method access)

    Returns:
        Tuple of (model_config, tokenizer, session)
    """
    original_config = base_service_class._get_model_config(model)
    if not original_config:
        raise ModelNotFoundError(f"Model '{model}' not found")

    # Resolve model path first (tests may patch this) and then validate
    # availability using the pre-fetched config and resolved path. This
    # avoids duplicate lookups and honors patched method calls.
    model_to_use_path = get_model_path(model, original_config, base_service_class)
    if not BaseNLPService._validate_model_availability(
        model,
        task="embedding",
        perform_filesystem_check=False,
        cfg=original_config,
        model_path=model_to_use_path,
    ):
        raise ModelNotFoundError(
            f"Model '{model}' not available or model files missing"
        )

    # Create a copy to avoid modifying the cached instance
    model_config = _copy_config(original_config)

    model_to_use_path = get_model_path(model, model_config, base_service_class)
    tokenizer = load_tokenizer(model_to_use_path, model_config, base_service_class)
    session = load_session(model_to_use_path, model_config, base_service_class)

    cache_stats = base_service_class._get_cache_stats()
    logger.info(
        f"Cache sizes - Encoder sessions: {cache_stats['encoder_sessions']}, "
        f"Model configs: {cache_stats['model_configs']}"
    )
    return model_config, tokenizer, session


def _copy_config(original_config: Any) -> Any:
    """Create a deep copy of model config to avoid modifying cached instance."""
    if hasattr(original_config, "model_copy"):
        return original_config.model_copy()
    elif hasattr(original_config, "copy"):
        return original_config.copy()
    else:
        # Fallback: create shallow copy using copy module
        return copy.copy(original_config)


def get_model_path(model: str, model_config: Any, base_service_class: Any) -> str:
    """Get validated model path.

    Args:
        model: Model name/identifier
        model_config: Model configuration object
        base_service_class: BaseNLPService class reference

    Returns:
        Validated absolute path to model directory
    """
    # Delegate to BaseNLPService model path resolver which prefers
    # `model_folder_name` and falls back to task-based folders.
    resolved = base_service_class._get_model_path(model)
    if not resolved:
        raise ModelLoadError(f"Failed to resolve model path for {model}")
    return resolved


def load_tokenizer(
    model_to_use_path: str, model_config: Any, base_service_class: Any
) -> Any:
    """Load and validate tokenizer.

    Args:
        model_to_use_path: Path to model directory
        model_config: Model configuration with tokenizer settings
        base_service_class: BaseNLPService class reference

    Returns:
        Loaded tokenizer instance
    """
    use_legacy = getattr(model_config, "legacy_tokenizer", False)
    tokenizer = base_service_class._get_tokenizer_threadsafe(
        model_to_use_path, use_legacy
    )
    if not tokenizer:
        raise TokenizerError(f"Failed to load tokenizer: {model_to_use_path}")
    return tokenizer


def load_session(
    model_to_use_path: str, model_config: Any, base_service_class: Any
) -> Any:
    """Load and validate ONNX session with fallback.

    Attempts to load optimized model first, falls back to regular model if needed.
    Auto-detects native dimension and output names from ONNX metadata.

    Args:
        model_to_use_path: Path to model directory
        model_config: Model configuration with ONNX settings
        base_service_class: BaseNLPService class reference

    Returns:
        Loaded ONNX Runtime session
    """
    # Use the base_service_class method to allow for mocking in tests
    model_path = base_service_class._get_embedding_model_path(
        model_to_use_path, model_config
    )

    try:
        session = base_service_class._get_encoder_session(model_path)
    except ModelLoadError as e:
        # If optimized model fails, try fallback to regular model
        use_optimized = getattr(model_config, "use_optimized", False)
        if use_optimized and "optimized" in str(e).lower():
            logger.warning(f"Optimized model failed: {e}, trying regular model")
            # Get regular model path
            regular_filename = (
                getattr(model_config, "encoder_onnx_model", None) or "model.onnx"
            )
            fallback_path = validate_safe_path(
                os.path.join(model_to_use_path, regular_filename),
                base_service_class._root_path,
            )
            session = base_service_class._get_encoder_session(fallback_path)
        else:
            raise e

    if not session:
        raise ModelLoadError(f"Failed to load ONNX session: {model_path}")

    # Auto-detect and validate native dimension
    _configure_dimension(session, model_config)

    # Auto-detect output names from ONNX model
    _configure_output_names(session, model_config)

    # Log cache stats
    from app.services.cache_registry import get_encoder_sessions

    encoder_sessions = get_encoder_sessions()
    logger.info(f"Encoder session cache size: {encoder_sessions.size()}")

    return session


def _configure_dimension(session: Any, model_config: Any) -> None:
    """Auto-detect native dimension from ONNX model and validate config."""
    native_dim = get_native_dimension_from_session(session)

    if native_dim:
        if not hasattr(model_config, "dimension") or model_config.dimension is None:
            # No dimension in config - use native
            model_config.dimension = native_dim
            logger.info(f"Auto-detected native dimension from ONNX model: {native_dim}")
        elif model_config.dimension > native_dim:
            # Config dimension is larger than native - use native instead
            original_dim = model_config.dimension
            model_config.dimension = native_dim
            logger.warning(
                f"Config dimension ({original_dim}) exceeds native dimension ({native_dim}). "
                f"Using native dimension to prevent upsampling."
            )
            # Store warning to be included in response
            if not hasattr(model_config, "_dimension_warning"):
                model_config._dimension_warning = (
                    f"Config dimension ({original_dim}) was larger than model's native dimension ({native_dim}). "
                    f"Using native dimension {native_dim} to avoid information loss."
                )


def _configure_output_names(session: Any, model_config: Any) -> None:
    """Auto-detect output names from ONNX model if not in config."""
    output_names_list = get_output_names_from_session(session)
    if output_names_list and (
        not hasattr(model_config, "outputnames") or not model_config.outputnames
    ):
        # Create outputnames object with primary output
        model_config.outputnames = SimpleNamespace(output=output_names_list[0])
        logger.info(f"Auto-detected primary output name: {output_names_list[0]}")


def get_embedding_model_path(
    model_to_use_path: str, model_config: Any, base_service_class: Any
) -> str:
    """Get the path to the embedding model file.

    Args:
        model_to_use_path: Path to model directory
        model_config: Model configuration with filename settings
        base_service_class: BaseNLPService class reference

    Returns:
        Validated path to ONNX model file
    """
    if not model_to_use_path:
        raise ModelLoadError("Model path is not provided to get_embedding_model_path")

    root = base_service_class._root_path
    if not root:
        raise ModelLoadError("Service root path is not configured")

    model_filename = get_model_filename(model_config, is_embedding=True)
    # Import here to allow test mocking via app.services.embedder_service
    from app.services import embedder_service

    return embedder_service.validate_safe_path(
        os.path.join(model_to_use_path, model_filename), root
    )


def get_model_filename(
    model_config: Any, is_embedding: bool = True, fallback_to_regular: bool = False
) -> str:
    """Delegate filename selection to BaseNLPService helper.

    Args:
        model_config: Model configuration
        is_embedding: Whether this is an embedding model (vs decoder)
        fallback_to_regular: Whether to fallback to regular model if optimized not found

    Returns:
        ONNX model filename
    """
    model_type = "encoder" if is_embedding else "decoder"
    return BaseNLPService._get_model_filename(
        model_config, model_type, fallback_to_regular
    )


def override_config_with_request(model_config: Any, request_params: dict) -> Any:
    """Use request parameters first, model config as fallback when parameters are None.

    Args:
        model_config: Base model configuration
        request_params: Request parameters to override config

    Returns:
        Updated model configuration with request overrides
    """
    override_fields = [
        "pooling_strategy",
        "max_length",
        "chunk_logic",
        "chunk_overlap",
        "chunk_size",
        "legacy_tokenizer",
        "normalize",
        "force_pooling",
        "lowercase",
        "remove_emojis",
        "use_optimized",
    ]

    for field in override_fields:
        if field in request_params:
            if request_params[field] is not None:
                # Use request parameter
                setattr(model_config, field, request_params[field])
                logger.debug(
                    f"Using request parameter for {field}: {request_params[field]}"
                )
            # If request param is None, keep model config value (fallback)
            else:
                logger.debug(
                    f"Using model config fallback for {field}: {getattr(model_config, field, None)}"
                )

    return model_config
