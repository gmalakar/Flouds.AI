# =============================================================================
# File: validate_utils.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Utility helpers for ONNX validation: pooling, normalization and comparisons.

This module extracts small pure-Python/numpy helpers to keep the main
validator script focused on orchestration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def mean_pooling(
    last_hidden_state: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    mask = mask[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    denom = mask.sum(axis=1)
    denom = np.maximum(denom, 1e-9)
    return summed / denom


def compare_arrays(ref: np.ndarray, onnx_arr: np.ndarray) -> dict:
    if ref.shape != onnx_arr.shape:
        return {
            "shape_mismatch": True,
            "ref_shape": ref.shape,
            "onnx_shape": onnx_arr.shape,
        }
    diff = np.abs(ref - onnx_arr)
    return {
        "shape_mismatch": False,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "l2": float(np.linalg.norm(ref - onnx_arr)),
    }


def l2_normalize(arr: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    try:
        norm = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)
        denom = np.maximum(norm, eps)
        return arr / denom
    except Exception:
        return arr


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity row-wise between two 2D arrays.

    Returns an array of shape (n_rows,) with cosine for each row.
    """
    ra = a.reshape((a.shape[0], -1))
    oa = b.reshape((b.shape[0], -1))
    rnorm = np.linalg.norm(ra, axis=1)
    onorm = np.linalg.norm(oa, axis=1)
    denom = rnorm * onorm
    denom = np.where(denom == 0, 1e-12, denom)
    return np.sum(ra * oa, axis=1) / denom
