# =============================================================================
# File: onnx_utils.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""ONNX model input/output preparation and name resolution."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.logger import get_logger
from app.services.embedder.models import DEFAULT_MAX_LENGTH

logger = get_logger("embedder_service.onnx")


def prepare_onnx_inputs(
    processed_text: str, tokenizer: Any, model_config: Any, session: Any
) -> Dict[str, Any]:
    """Prepare ONNX inputs for inference with auto-detected input names.

    Args:
        processed_text: Preprocessed text ready for tokenization
        tokenizer: Loaded tokenizer instance
        model_config: Model configuration object
        session: ONNX Runtime session

    Returns:
        Dictionary mapping input names to numpy arrays
    """
    max_length = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)

    encoding = tokenizer(
        processed_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask")

    # Ensure correct dtypes
    if input_ids.dtype != np.int64:
        input_ids = input_ids.astype(np.int64)
    if attention_mask is not None and attention_mask.dtype != np.int64:
        attention_mask = attention_mask.astype(np.int64)

    def _ensure_batch_dim(arr):
        """Ensure array has batch dimension as first axis."""
        if arr.ndim == 1:
            return arr[None, :]
        return arr

    input_ids = _ensure_batch_dim(input_ids)
    if attention_mask is not None:
        attention_mask = _ensure_batch_dim(attention_mask)

    # Auto-detect input names from ONNX model or use config
    model_input_names = [inp.name for inp in session.get_inputs()]
    input_names_config = getattr(model_config, "inputnames", {})

    def _select_name(
        config_name: Optional[str],
        candidates: List[str],
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Select input name using 4-tier matching: config → exact → case-insensitive → substring."""
        # 1. Config override
        if config_name:
            return config_name

        # 2. Exact match
        for cand in candidates:
            if cand in model_input_names:
                return cand

        # 3. Case-insensitive match
        model_input_names_lc = [n.lower() for n in model_input_names]
        for cand in candidates:
            lc = cand.lower()
            if lc in model_input_names_lc:
                return model_input_names[model_input_names_lc.index(lc)]

        # 4. Substring match (fallback)
        for cand in candidates:
            lc = cand.lower()
            for name, name_lc in zip(model_input_names, model_input_names_lc):
                if lc in name_lc:
                    return name

        return default

    # Build inputs dictionary with auto-detected names
    input_id_name = _select_name(
        getattr(input_names_config, "input", None),
        ["input_ids", "input"],
        "input_ids",
    )
    inputs: Dict[str, Any] = {str(input_id_name): input_ids}

    if attention_mask is not None:
        mask_name = _select_name(
            getattr(input_names_config, "mask", None),
            ["attention_mask", "mask"],
            None,
        )
        if mask_name:
            inputs[str(mask_name)] = attention_mask

    # Add optional inputs if model requires them
    seq_length = input_ids.shape[1] if input_ids.ndim > 1 else len(input_ids)
    _add_optional_inputs(
        inputs, input_names_config, model_input_names, session, seq_length
    )

    return inputs


def _find_matching_input(
    name: str, model_input_names: List[str], case_sensitive: bool = True
) -> Optional[str]:
    """Find matching input name in model, with optional case-insensitive search."""
    if case_sensitive and name in model_input_names:
        return name

    name_lower = name.lower()
    for model_name in model_input_names:
        if model_name.lower() == name_lower:
            return model_name
    return None


def _add_optional_inputs(
    inputs: Dict[str, Any],
    input_names_config: Any,
    model_input_names: List[str],
    session: Any,
    seq_length: Optional[int] = None,
) -> None:
    """Add optional inputs like position_ids, token_type_ids if model requires them.

    Args:
        inputs: Dictionary to add optional inputs to
        input_names_config: Config object with input name mappings
        model_input_names: List of model input names (can be Mock in tests)
        session: ONNX session (can be Mock in tests)
        seq_length: Sequence length for generating position_ids and token_type_ids
    """
    # Extract model input names from session
    if hasattr(session, "get_inputs") and callable(session.get_inputs):
        try:
            get_inputs_result = session.get_inputs()
            # Check if it returns an iterable with items that have .name attribute
            if hasattr(get_inputs_result, "__iter__"):
                actual_model_input_names = [inp.name for inp in get_inputs_result]
            else:
                # Mock or invalid result
                actual_model_input_names = []
        except Exception:
            actual_model_input_names = []
    elif isinstance(model_input_names, list):
        actual_model_input_names = model_input_names
    else:
        # Fallback: empty list
        actual_model_input_names = []

    actual_seq_length = seq_length if seq_length is not None else 128

    # Get input_ids shape to determine sequence length if not provided directly
    input_ids = inputs.get("input_ids")
    if (
        input_ids is not None
        and hasattr(input_ids, "shape")
        and hasattr(input_ids, "ndim")
    ):
        try:
            detected_seq_length = (
                input_ids.shape[1] if input_ids.ndim > 1 else len(input_ids)
            )
        except (TypeError, AttributeError):
            detected_seq_length = actual_seq_length
    else:
        detected_seq_length = actual_seq_length

    # Position IDs
    position_ids_name = getattr(input_names_config, "position_ids", None)
    if not isinstance(position_ids_name, str):
        position_ids_name = getattr(input_names_config, "position", None)
    if not isinstance(position_ids_name, str):
        position_ids_name = None
    if not position_ids_name:
        position_ids_name = _find_matching_input(
            "position_ids", actual_model_input_names, case_sensitive=False
        )
    if position_ids_name:
        position_ids = np.arange(detected_seq_length, dtype=np.int64)[None, :]
        inputs[str(position_ids_name)] = position_ids

    # Token type IDs
    token_type_ids_name = getattr(input_names_config, "token_type_ids", None)
    if not isinstance(token_type_ids_name, str):
        token_type_ids_name = getattr(input_names_config, "tokentype", None)
    if not isinstance(token_type_ids_name, str):
        token_type_ids_name = None
    if not token_type_ids_name:
        token_type_ids_name = _find_matching_input(
            "token_type_ids", actual_model_input_names, case_sensitive=False
        )
    if token_type_ids_name:
        if input_ids is not None and hasattr(input_ids, "shape"):
            try:
                token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            except (TypeError, AttributeError):
                token_type_ids = np.zeros((1, detected_seq_length), dtype=np.int64)
        else:
            token_type_ids = np.zeros((1, detected_seq_length), dtype=np.int64)
        inputs[str(token_type_ids_name)] = token_type_ids

    # Decoder input IDs (for encoder-decoder models)
    decoder_input_ids_name = getattr(input_names_config, "decoder_input_ids", None)
    if not decoder_input_ids_name:
        decoder_input_ids_name = _find_matching_input(
            "decoder_input_ids", actual_model_input_names, case_sensitive=False
        )
    if decoder_input_ids_name:
        if input_ids is not None and hasattr(input_ids, "shape"):
            try:
                decoder_input_ids = np.zeros((input_ids.shape[0], 1), dtype=np.int64)
            except (TypeError, AttributeError):
                decoder_input_ids = np.zeros((1, 1), dtype=np.int64)
        else:
            decoder_input_ids = np.zeros((1, 1), dtype=np.int64)
        inputs[str(decoder_input_ids_name)] = decoder_input_ids


def log_onnx_outputs(outputs: List[np.ndarray], session: Any) -> None:
    """Log ONNX output tensor information for debugging."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    try:
        output_names = [out.name for out in session.get_outputs()]
        for idx, (output, name) in enumerate(zip(outputs, output_names)):
            logger.debug(
                f"ONNX output {idx} ({name}): shape={output.shape}, dtype={output.dtype}"
            )
    except Exception as e:
        logger.debug(f"Could not log ONNX outputs: {e}")


def get_native_dimension_from_session(session: Any) -> Optional[int]:
    """Extract the native embedding dimension from ONNX session output shape.

    Args:
        session: ONNX Runtime session

    Returns:
        Native dimension from last axis of first output, or None if not detected
    """
    try:
        outputs = session.get_outputs()
        if outputs and len(outputs) > 0:
            output_shape = outputs[0].shape
            # Output shape is typically ['batch_size', 'sequence_length', dimension]
            # The last dimension is the embedding dimension
            if output_shape and len(output_shape) >= 3:
                # Handle symbolic dimensions (e.g., 'batch_size') vs numeric
                last_dim = output_shape[-1]
                if isinstance(last_dim, (int, np.integer)):
                    logger.debug(
                        f"Detected native dimension from ONNX output: {last_dim}"
                    )
                    return int(last_dim)
    except Exception as e:
        logger.warning(f"Could not auto-detect dimension from ONNX session: {e}")
    return None


def get_output_names_from_session(session: Any) -> List[str]:
    """Extract output tensor names from ONNX session.

    Args:
        session: ONNX Runtime session

    Returns:
        List of output tensor names
    """
    try:
        outputs = session.get_outputs()
        if outputs:
            output_names = [output.name for output in outputs]
            logger.debug(f"Auto-detected output names from ONNX model: {output_names}")
            return output_names
    except Exception as e:
        logger.warning(f"Could not auto-detect output names from ONNX session: {e}")
    return []
