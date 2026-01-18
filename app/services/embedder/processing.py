# =============================================================================
# File: processing.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Embedding output processing, pooling, projection, and quantization."""

import logging
from typing import Any, Optional, cast

import numpy as np
from numpy import ndarray

from app.logger import get_logger
from app.utils.constants import NORM_EPS, RANDOM_SEED
from app.utils.pooling_strategies import PoolingStrategies

logger = get_logger("embedder_service.processing")


def process_embedding_output(
    hidden_states: ndarray,
    model_config: Any,
    attention_mask: Optional[ndarray],
    projected_dimension: Optional[int],
    **kwargs: Any,
) -> ndarray:
    """Process ONNX model output into final embedding vector.

    Applies pooling, projection (dimensionality reduction), and normalization.

    Args:
        hidden_states: Raw ONNX output (batch_size, seq_len, hidden_dim)
        model_config: Model configuration with pooling settings
        attention_mask: Attention mask for pooling
        projected_dimension: Target dimension for projection (optional)
        **kwargs: Additional parameters including prefetched_proj_matrix

    Returns:
        Processed embedding vector as numpy array
    """
    # Apply pooling
    pooling_strategy = getattr(model_config, "pooling_strategy", "mean")
    force_pooling = getattr(model_config, "force_pooling", False)

    # Normalize mask shape if provided
    if attention_mask is not None:
        attention_mask = _normalize_mask(attention_mask, hidden_states)

    _pooled_res: Optional[ndarray] = PoolingStrategies.apply(
        hidden_states,
        pooling_strategy,
        attention_mask,
        force_pooling,
    )
    # Ensure we always have a numpy ndarray to satisfy static typing
    if _pooled_res is None:
        pooled = np.zeros((hidden_states.shape[0], hidden_states.shape[-1]), dtype=np.float32)
    else:
        pooled = cast(ndarray, _pooled_res)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"After pooling: {pooled.shape}")

    # Apply projection if requested
    if projected_dimension and projected_dimension > 0:
        native_dimension = getattr(model_config, "dimension", None)
        # If native_dimension not in config or doesn't match actual size, use actual embedding dimension
        actual_dimension = pooled.shape[-1]
        if native_dimension is None or native_dimension != actual_dimension:
            native_dimension = actual_dimension

        if native_dimension and projected_dimension < native_dimension:
            pooled = project_embedding(
                pooled,
                native_dimension,
                projected_dimension,
                kwargs.get("prefetched_proj_matrix"),
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"After projection: {pooled.shape}")

    # Normalize if requested
    normalize = getattr(model_config, "normalize", True)
    if normalize:
        pooled = _normalize_vector(pooled)

    # Apply quantization if requested
    quantize = kwargs.get("quantize", getattr(model_config, "quantize", False))
    if quantize:
        quantize_type = kwargs.get("quantize_type", getattr(model_config, "quantize_type", "int8"))
        pooled = quantize_embedding(pooled, quantize_type)

    return pooled


def _normalize_mask(attention_mask: ndarray, hidden_states: ndarray) -> ndarray:
    """Normalize attention mask to match hidden states shape.

    Args:
        attention_mask: Attention mask (various shapes)
        hidden_states: Hidden states tensor (batch, seq_len, dim)

    Returns:
        Normalized mask matching hidden states batch and sequence dimensions
    """
    if attention_mask.ndim == 1:
        # (seq_len,) -> (1, seq_len)
        attention_mask = attention_mask[None, :]

    if attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
        # (batch, 1, seq_len) -> (batch, seq_len)
        attention_mask = attention_mask.squeeze(1)

    # Ensure batch dimension matches
    if attention_mask.shape[0] != hidden_states.shape[0]:
        # Broadcast to match batch size
        attention_mask = np.broadcast_to(
            attention_mask, (hidden_states.shape[0], attention_mask.shape[-1])
        )

    return attention_mask


def _normalize_vector(embedding: ndarray) -> ndarray:
    """L2 normalize embedding vector."""
    norm = np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True)
    return embedding / np.maximum(norm, NORM_EPS)


def project_embedding(
    embedding: ndarray,
    native_dimension: int,
    projected_dimension: int,
    prefetched_proj_matrix: Optional[ndarray] = None,
) -> ndarray:
    """Apply dimensionality reduction via random projection.

    Args:
        embedding: Input embedding vector
        native_dimension: Original embedding dimension
        projected_dimension: Target reduced dimension
        prefetched_proj_matrix: Pre-computed projection matrix (optional)

    Returns:
        Projected embedding vector
    """
    proj_matrix: Optional[ndarray] = None
    if prefetched_proj_matrix is not None:
        proj_matrix = prefetched_proj_matrix
    else:
        # Try to get from cache first
        try:
            from app.services.cache_registry import get_projection_matrix_cache

            cache_key = f"proj:{native_dimension}x{projected_dimension}"
            proj_cache = get_projection_matrix_cache()
            proj_matrix = proj_cache.get(cache_key)

            if proj_matrix is None:
                # Generate projection matrix with deterministic seed
                rng = np.random.default_rng(seed=RANDOM_SEED)
                proj_matrix = rng.uniform(-1, 1, (native_dimension, projected_dimension)).astype(
                    np.float32
                )
                # Cache it for future use
                proj_cache.put(cache_key, proj_matrix)
        except Exception:
            # Fallback: generate without caching
            rng = np.random.default_rng(seed=RANDOM_SEED)
            proj_matrix = rng.uniform(-1, 1, (native_dimension, projected_dimension)).astype(
                np.float32
            )

    # Apply projection: (batch, native_dim) @ (native_dim, proj_dim) -> (batch, proj_dim)
    proj_matrix = cast(ndarray, proj_matrix)
    projected = embedding @ proj_matrix
    return projected


def quantize_embedding(embedding: ndarray, quantization_type: str) -> ndarray:
    """Quantize embedding vector for storage efficiency.

    Args:
        embedding: Input embedding vector (float32)
        quantization_type: Type of quantization - "binary", "int8", "uint8", or "ubinary"

    Returns:
        Quantized embedding vector
    """
    # Handle empty arrays
    if embedding.size == 0:
        if quantization_type == "binary":
            return embedding.astype(np.int8)
        elif quantization_type == "int8":
            return embedding.astype(np.int8)
        elif quantization_type == "uint8":
            return embedding.astype(np.uint8)
        else:
            return embedding

    if quantization_type == "binary":
        # Binary quantization: -1 for negative/zero, 1 for positive
        result = np.where(embedding > 0, 1, -1).astype(np.int8)
        return result
    elif quantization_type == "int8":
        # Scale to int8 range [-127, 127] (avoid -128)
        # Assumes embedding is already normalized (e.g., [-1, 1] or similar)
        scaled = embedding * 127.0
        # Clip to [-127, 127] to avoid -128
        clipped = np.clip(scaled, -127, 127)
        return clipped.astype(np.int8)
    elif quantization_type == "uint8":
        # Scale to uint8 range [0, 255] with per-sample min-max scaling
        is_batch = embedding.ndim > 1
        if is_batch:
            # Per-row scaling for batch
            min_vals = embedding.min(axis=1, keepdims=True)
            max_vals = embedding.max(axis=1, keepdims=True)
            # Handle zero-range rows
            ranges = max_vals - min_vals
            ranges = np.where(ranges > 0, ranges, 1.0)
            scaled = (embedding - min_vals) / ranges * 255.0
        else:
            # Single vector scaling
            min_val = embedding.min()
            max_val = embedding.max()
            if max_val - min_val > 0:
                scaled = (embedding - min_val) / (max_val - min_val) * 255.0
            else:
                scaled = np.full_like(embedding, 128.0)  # Middle value for constant
        return np.clip(scaled, 0, 255).astype(np.uint8)
    elif quantization_type == "ubinary":
        # Unsigned binary: threshold at mean
        threshold = embedding.mean()
        return (embedding > threshold).astype(np.uint8)
    else:
        # Unknown type, return as-is
        return embedding


def get_prefetched_proj_matrix(
    model_config: Any, projected_dimension: Optional[int]
) -> Optional[ndarray]:
    """Prefetch projection matrix from cache or generate new one.

    Args:
        model_config: Model configuration with native dimension
        projected_dimension: Target projection dimension

    Returns:
        Projection matrix (native_dim x proj_dim) or None if not needed
    """
    try:
        native_dim = getattr(model_config, "dimension", None)
        if (
            projected_dimension is None
            or projected_dimension <= 0
            or native_dim is None
            or projected_dimension >= native_dim
        ):
            return None

        from app.services.cache_registry import get_projection_matrix_cache

        cache_key = f"proj:{native_dim}x{projected_dimension}"
        proj_cache = get_projection_matrix_cache()
        mat = proj_cache.get(cache_key)
        if mat is None:
            rng = np.random.default_rng(seed=RANDOM_SEED)
            mat = rng.uniform(-1, 1, (native_dim, projected_dimension)).astype(np.float32)
            proj_cache.put(cache_key, mat)
        return mat
    except Exception:
        return None
