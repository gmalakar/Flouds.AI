# =============================================================================
# File: config.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Configuration resolution for prompt service models."""

import os
from typing import Any, Dict, Optional

from app.config.onnx_config import OnnxConfig
from app.exceptions import TokenizerError
from app.logger import get_logger
from app.services.base_nlp_service import BaseNLPService
from app.utils.path_validator import validate_safe_path

logger = get_logger(__name__)


def get_model_config(model_to_use: str) -> Optional[OnnxConfig]:
    """Get ONNX model configuration for a model name.

    Delegates to BaseNLPService which loads from the centralized config.

    Args:
        model_to_use: Model name (e.g., 't5-small')

    Returns:
        OnnxConfig object, or None if not found
    """
    return BaseNLPService._get_model_config(model_to_use)


def get_model_path(model_to_use: str) -> Optional[str]:
    """Get filesystem path to model directory.

    Delegates to BaseNLPService which resolves from config and environment.

    Args:
        model_to_use: Model name

    Returns:
        Absolute path to model directory, or None if resolution failed
    """
    return BaseNLPService._get_model_path(model_to_use)


def get_model_filename(model_config: OnnxConfig, model_type: str) -> str:
    """Get model filename based on type.

    Args:
        model_config: ONNX configuration
        model_type: 'encoder' or 'decoder'

    Returns:
        Filename (e.g., 'encoder_model.onnx' or 'decoder_model.onnx')
    """
    if model_type == "encoder":
        return model_config.encoder_onnx_model or "encoder_model.onnx"
    else:  # decoder
        return model_config.decoder_onnx_model or "decoder_model.onnx"


def get_model_file_paths(model_path: str, model_config: OnnxConfig) -> tuple[str, str]:
    """Get encoder and decoder model file paths.

    Args:
        model_path: Root model directory path
        model_config: ONNX configuration

    Returns:
        Tuple of (encoder_path, decoder_path) after path validation
    """
    encoder_filename = get_model_filename(model_config, "encoder")
    decoder_filename = get_model_filename(model_config, "decoder")

    root_path = BaseNLPService._root_path
    encoder_path = validate_safe_path(
        os.path.join(model_path, encoder_filename), root_path
    )
    decoder_path = validate_safe_path(
        os.path.join(model_path, decoder_filename), root_path
    )

    logger.debug(f"Encoder path: {encoder_path}, Decoder path: {decoder_path}")
    return encoder_path, decoder_path


def get_tokenizer_threadsafe(model_path: str, use_legacy: bool = False) -> Any:
    """Get tokenizer for a model, with thread-safe loading.

    Delegates to BaseNLPService for consistent tokenizer handling.

    Args:
        model_path: Path to model directory
        use_legacy: Whether to use legacy tokenizer

    Returns:
        Loaded tokenizer instance

    Raises:
        TokenizerError: If tokenizer loading fails
    """
    tokenizer = BaseNLPService._get_tokenizer_threadsafe(model_path, use_legacy)
    if not tokenizer:
        raise TokenizerError(f"Failed to load tokenizer: {model_path}")
    return tokenizer


def get_token_config(model_config: OnnxConfig, tokenizer: Any) -> Dict[str, int]:
    """Get token configuration for ONNX inference.

    Args:
        model_config: ONNX configuration
        tokenizer: Loaded tokenizer instance

    Returns:
        Dictionary with pad_token_id, eos_token_id, decoder_start_token_id, max_length
    """
    from app.services.prompt.models import DEFAULT_MAX_LENGTH

    pad_token_id = getattr(
        model_config, "pad_token_id", getattr(tokenizer, "pad_token_id", 0)
    )
    eos_token_id = getattr(
        model_config, "eos_token_id", getattr(tokenizer, "eos_token_id", 1)
    )
    max_length = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)

    # Get decoder start token
    decoder_start_token_id = (
        getattr(model_config, "decoder_start_token_id", None)
        or getattr(tokenizer, "decoder_start_token_id", None)
        or getattr(tokenizer, "bos_token_id", None)
        or pad_token_id
    )

    return {
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "decoder_start_token_id": decoder_start_token_id,
        "max_length": max_length,
    }


def validate_model_availability(
    model_name: str,
    task: str = "prompt",
    perform_filesystem_check: bool = False,
    cfg: Optional[OnnxConfig] = None,
    model_path: Optional[str] = None,
) -> bool:
    """Validate that a model is available for use.

    Delegates to BaseNLPService for consistency.

    Args:
        model_name: Model name
        task: Task type (default: 'prompt')
        perform_filesystem_check: Whether to verify files exist
        cfg: Optional pre-loaded OnnxConfig
        model_path: Optional pre-resolved model path

    Returns:
        True if model is available, False otherwise
    """
    return BaseNLPService._validate_model_availability(
        model_name,
        task=task,
        perform_filesystem_check=perform_filesystem_check,
        cfg=cfg,
        model_path=model_path,
    )
