# =============================================================================
# File: test_normalization_performance.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_normalization_performance.py
# Test optimized normalization implementation
# =============================================================================

import numpy as np
import pytest  # noqa: F401

from app.utils.constants import NORM_EPS


def normalize_original(embedding):
    """Original normalization using np.linalg.norm."""
    if embedding.ndim == 1:
        norm = np.linalg.norm(embedding)
        if norm > NORM_EPS:
            return embedding / norm
    elif embedding.ndim == 2:
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norms_safe = np.where(row_norms > NORM_EPS, row_norms, 1.0)
        return embedding / row_norms_safe
    else:
        norm = np.linalg.norm(embedding)
        if norm > NORM_EPS:
            return embedding / norm
    return embedding


def normalize_optimized(embedding):
    """Optimized normalization using element-wise operations."""
    if embedding.ndim == 1:
        norm = np.sqrt(np.sum(embedding * embedding))
        if norm > NORM_EPS:
            return embedding / norm
    elif embedding.ndim == 2:
        row_norms = np.sqrt(np.sum(embedding * embedding, axis=1, keepdims=True))
        row_norms_safe = np.where(row_norms > NORM_EPS, row_norms, 1.0)
        return embedding / row_norms_safe
    else:
        norm = np.sqrt(np.sum(embedding * embedding))
        if norm > NORM_EPS:
            return embedding / norm
    return embedding


class TestNormalizationOptimization:
    """Test that optimized normalization produces identical results."""

    def test_1d_normalization_equivalence(self):
        """Test 1D vector normalization produces same results."""
        vec = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)

        result_orig = normalize_original(vec.copy())
        result_opt = normalize_optimized(vec.copy())

        np.testing.assert_allclose(result_orig, result_opt, rtol=1e-6)

        # Verify it's normalized (unit length)
        assert abs(np.linalg.norm(result_opt) - 1.0) < 1e-6

    def test_2d_normalization_equivalence(self):
        """Test 2D batch normalization produces same results."""
        batch = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)

        result_orig = normalize_original(batch.copy())
        result_opt = normalize_optimized(batch.copy())

        np.testing.assert_allclose(result_orig, result_opt, rtol=1e-6)

        # Verify each row is normalized
        for row in result_opt:
            assert abs(np.linalg.norm(row) - 1.0) < 1e-6

    def test_zero_vector_handling(self):
        """Test that zero vectors are handled correctly."""
        zero_vec = np.zeros(5, dtype=np.float32)

        result_orig = normalize_original(zero_vec.copy())
        result_opt = normalize_optimized(zero_vec.copy())

        np.testing.assert_array_equal(result_orig, result_opt)
        np.testing.assert_array_equal(result_opt, zero_vec)

    def test_near_zero_vector_handling(self):
        """Test vectors with very small norms."""
        tiny_vec = np.array([1e-15, 1e-15, 1e-15], dtype=np.float32)

        result_orig = normalize_original(tiny_vec.copy())
        result_opt = normalize_optimized(tiny_vec.copy())

        # Both should leave it unchanged (below epsilon threshold)
        np.testing.assert_allclose(result_orig, result_opt, rtol=1e-6)

    def test_batch_with_zero_rows(self):
        """Test batch normalization with some zero rows."""
        batch = np.array(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],  # Zero row
            dtype=np.float32,
        )

        result_orig = normalize_original(batch.copy())
        result_opt = normalize_optimized(batch.copy())

        np.testing.assert_allclose(result_orig, result_opt, rtol=1e-6)

        # First and third rows should be normalized, second should be zero
        assert abs(np.linalg.norm(result_opt[0]) - 1.0) < 1e-6
        np.testing.assert_array_equal(result_opt[1], np.zeros(3))
        assert abs(np.linalg.norm(result_opt[2]) - 1.0) < 1e-6

    def test_large_values(self):
        """Test normalization with large values."""
        large_vec = np.array([1e6, 2e6, 3e6], dtype=np.float32)

        result_orig = normalize_original(large_vec.copy())
        result_opt = normalize_optimized(large_vec.copy())

        np.testing.assert_allclose(result_orig, result_opt, rtol=1e-6)
        assert abs(np.linalg.norm(result_opt) - 1.0) < 1e-5

    def test_maintains_float32_dtype(self):
        """Test that float32 dtype is preserved."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = normalize_optimized(vec)
        assert result.dtype == np.float32

    def test_mixed_sign_values(self):
        """Test normalization with negative values."""
        vec = np.array([-3.0, 4.0, -5.0, 6.0], dtype=np.float32)

        result_orig = normalize_original(vec.copy())
        result_opt = normalize_optimized(vec.copy())

        np.testing.assert_allclose(result_orig, result_opt, rtol=1e-6)
        assert abs(np.linalg.norm(result_opt) - 1.0) < 1e-6
