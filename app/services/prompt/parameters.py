# =============================================================================
# File: parameters.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Generation parameter building utilities for prompt service."""

from typing import Any, Dict

from app.config.onnx_config import OnnxConfig
from app.logger import get_logger
from app.models.prompt_request import PromptRequest
from app.services.prompt.models import DEFAULT_MAX_LENGTH

logger = get_logger(__name__)


def build_generation_params(model_config: OnnxConfig, request: PromptRequest) -> Dict[str, Any]:
    """Build generation parameters for model.

    Args:
        model_config: ONNX model configuration
        request: Prompt request with generation parameters

    Returns:
        Dictionary of generation parameters for the model
    """
    generate_kwargs: Dict[str, Any] = {}

    # Basic parameters
    add_basic_params(generate_kwargs, model_config)
    add_beam_params(generate_kwargs, model_config)
    add_optional_params(generate_kwargs, model_config)
    add_temperature_params(generate_kwargs, model_config, request)

    logger.debug(f"Generation parameters: {generate_kwargs}")
    return generate_kwargs


def add_basic_params(generate_kwargs: Dict[str, Any], model_config: OnnxConfig) -> None:
    """Add basic generation parameters (max_length, min_length).

    Args:
        generate_kwargs: Dictionary to add parameters to
        model_config: ONNX model configuration
    """
    for param, default in [("max_length", DEFAULT_MAX_LENGTH), ("min_length", 0)]:
        value = getattr(model_config, param, default)
        if value is not None:
            generate_kwargs[param] = value


def add_beam_params(generate_kwargs: Dict[str, Any], model_config: OnnxConfig) -> None:
    """Add beam search parameters.

    Args:
        generate_kwargs: Dictionary to add parameters to
        model_config: ONNX model configuration
    """
    num_beams = getattr(model_config, "num_beams", 1)
    if num_beams > 1:
        generate_kwargs["num_beams"] = num_beams
    if getattr(model_config, "early_stopping", False):
        generate_kwargs["early_stopping"] = True


def add_optional_params(generate_kwargs: Dict[str, Any], model_config: OnnxConfig) -> None:
    """Add optional generation parameters (repetition_penalty, top_k, etc.).

    Args:
        generate_kwargs: Dictionary to add parameters to
        model_config: ONNX model configuration
    """
    for param in [
        "repetition_penalty",
        "length_penalty",
        "top_k",
        "top_p",
        "no_repeat_ngram_size",
    ]:
        value = getattr(model_config, param, None)
        if value is not None:
            generate_kwargs[param] = value


def add_temperature_params(
    generate_kwargs: Dict[str, Any],
    model_config: OnnxConfig,
    request: PromptRequest,
) -> None:
    """Add temperature and sampling parameters.

    Args:
        generate_kwargs: Dictionary to add parameters to
        model_config: ONNX model configuration
        request: Prompt request with temperature setting
    """
    temp_src = (
        request.temperature
        if request.temperature is not None
        else getattr(model_config, "temperature", 0.0)
    )
    try:
        temperature_val = float(temp_src)
    except Exception:
        temperature_val = 0.0

    if temperature_val > 0.0:
        generate_kwargs["temperature"] = temperature_val
        generate_kwargs["do_sample"] = True
