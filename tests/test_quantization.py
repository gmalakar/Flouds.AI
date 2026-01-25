# =============================================================================
# File: test_quantization.py
# Date: 2026-01-06
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import numpy as np
import pytest  # noqa: F401

from app.services.embedder import SentenceTransformer


class TestQuantization:
    """Test suite for embedding quantization functionality."""

    def test_int8_quantization_single_vector(self):
        """Test int8 quantization for single normalized vector."""
        embedding = np.array([0.5, -0.3, 0.0, 1.0, -1.0], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "int8")

        assert quantized.dtype == np.int8
        assert quantized.shape == embedding.shape
        # Check values are in expected range
        assert np.all(quantized >= -127)
        assert np.all(quantized <= 127)
        # Check specific mappings
        assert quantized[3] == 127  # 1.0 -> 127
        assert quantized[4] == -127  # -1.0 -> -127
        assert quantized[2] == 0  # 0.0 -> 0

    def test_int8_quantization_batch(self):
        """Test int8 quantization for batch of normalized vectors."""
        embeddings = np.array(
            [[0.5, -0.5, 0.0], [1.0, 0.0, -1.0], [-0.25, 0.75, 0.1]], dtype=np.float32
        )
        quantized = SentenceTransformer._quantize_embedding(embeddings, "int8")

        assert quantized.dtype == np.int8
        assert quantized.shape == embeddings.shape
        assert np.all(quantized >= -127)
        assert np.all(quantized <= 127)

    def test_uint8_quantization_single_vector(self):
        """Test uint8 quantization with min-max scaling."""
        embedding = np.array([1.5, 2.0, 1.0, 1.75, 1.25], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "uint8")

        assert quantized.dtype == np.uint8
        assert quantized.shape == embedding.shape
        assert np.all(quantized >= 0)
        assert np.all(quantized <= 255)
        # Min value should map to 0, max to 255
        assert quantized[2] == 0  # min value
        assert quantized[1] == 255  # max value

    def test_uint8_quantization_batch(self):
        """Test uint8 quantization for batch (per-row scaling)."""
        embeddings = np.array(
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [-5.0, 0.0, 5.0]], dtype=np.float32
        )
        quantized = SentenceTransformer._quantize_embedding(embeddings, "uint8")

        assert quantized.dtype == np.uint8
        assert quantized.shape == embeddings.shape
        assert np.all(quantized >= 0)
        assert np.all(quantized <= 255)
        # Each row should have min=0 and max=255
        assert np.all(quantized.min(axis=1) == 0)
        assert np.all(quantized.max(axis=1) == 255)

    def test_binary_quantization_single_vector(self):
        """Test binary quantization (sign-based thresholding)."""
        embedding = np.array([0.5, -0.3, 0.0, 0.001, -0.001], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "binary")

        assert quantized.dtype == np.int8
        assert quantized.shape == embedding.shape
        # All values should be either -1 or 1
        assert np.all(np.isin(quantized, [-1, 1]))
        # Check specific mappings
        assert quantized[0] == 1  # positive -> 1
        assert quantized[1] == -1  # negative -> -1
        assert quantized[2] == -1  # zero -> -1 (threshold)
        assert quantized[3] == 1  # small positive -> 1
        assert quantized[4] == -1  # small negative -> -1

    def test_binary_quantization_batch(self):
        """Test binary quantization for batch."""
        embeddings = np.array(
            [[0.1, -0.2, 0.3], [-1.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        quantized = SentenceTransformer._quantize_embedding(embeddings, "binary")

        assert quantized.dtype == np.int8
        assert quantized.shape == embeddings.shape
        assert np.all(np.isin(quantized, [-1, 1]))

    def test_quantization_preserves_shape(self):
        """Test that quantization preserves tensor shape for various dimensions."""
        shapes = [(128,), (10, 128), (32, 384), (1, 1536)]
        for shape in shapes:
            embedding = np.random.randn(*shape).astype(np.float32)
            for qtype in ["int8", "uint8", "binary"]:
                quantized = SentenceTransformer._quantize_embedding(embedding, qtype)
                assert quantized.shape == shape, f"Shape mismatch for {qtype} with shape {shape}"

    def test_int8_roundtrip_accuracy(self):
        """Test int8 quantization preserves approximate values after dequantization."""
        embedding = np.array([0.0, 0.25, 0.5, 0.75, -0.5, -1.0], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "int8")

        # Dequantize back to float
        dequantized = quantized.astype(np.float32) / 127.0

        # Check relative error is small (within quantization step size)
        max_error = np.abs(embedding - dequantized).max()
        assert max_error < 0.01, f"Dequantization error too large: {max_error}"

    def test_uint8_zero_range_handling(self):
        """Test uint8 quantization handles constant vectors (zero range)."""
        # All values the same
        embedding = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "uint8")

        assert quantized.dtype == np.uint8
        # Should handle gracefully without division by zero
        assert not np.any(np.isnan(quantized))
        assert not np.any(np.isinf(quantized))

    def test_uint8_batch_zero_range_handling(self):
        """Test uint8 quantization handles constant rows in batch."""
        embeddings = np.array(
            [[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [-1.0, 0.0, 1.0]], dtype=np.float32
        )
        quantized = SentenceTransformer._quantize_embedding(embeddings, "uint8")

        assert quantized.dtype == np.uint8
        assert not np.any(np.isnan(quantized))
        # Row with constant values should not cause issues
        assert quantized[1, 0] == quantized[1, 1] == quantized[1, 2]

    def test_binary_maximizes_compression(self):
        """Test binary quantization achieves maximum compression (1 bit per value)."""
        embedding = np.random.randn(1536).astype(np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "binary")

        # Binary only uses 2 distinct values
        unique_values = np.unique(quantized)
        assert len(unique_values) <= 2
        assert np.all(np.isin(unique_values, [-1, 1]))

    def test_unknown_quantization_type_returns_original(self):
        """Test that unknown quantization type returns original embedding."""
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "unknown_type")

        assert quantized.dtype == np.float32
        assert np.array_equal(quantized, embedding)

    def test_int8_clipping_at_boundaries(self):
        """Test int8 quantization clips values outside [-1, 1] range."""
        # Values outside normalized range
        embedding = np.array([-2.0, -1.5, 1.5, 2.0], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "int8")

        # Should clip to int8 range
        assert quantized.min() >= -127
        assert quantized.max() <= 127
        # Extreme values should be clipped
        assert quantized[0] == -127  # -2.0 * 127 = -254 -> clipped to -127
        assert quantized[3] == 127  # 2.0 * 127 = 254 -> clipped to 127

    def test_quantization_storage_reduction(self):
        """Test quantization actually reduces storage size."""
        embedding = np.random.randn(1536).astype(np.float32)

        float32_size = embedding.nbytes
        int8_size = SentenceTransformer._quantize_embedding(embedding, "int8").nbytes
        uint8_size = SentenceTransformer._quantize_embedding(embedding, "uint8").nbytes
        binary_size = SentenceTransformer._quantize_embedding(embedding, "binary").nbytes

        # int8 and uint8 should be 4x smaller than float32
        assert int8_size == float32_size // 4
        assert uint8_size == float32_size // 4
        # binary still stored as int8, but logically uses 1 bit per value
        assert binary_size == float32_size // 4

    def test_batch_quantization_independence(self):
        """Test that batch quantization processes each row independently (uint8)."""
        embeddings = np.array([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embeddings, "uint8")

        # Each row should be independently scaled to [0, 255]
        # Row 0: min=1.0->0, max=3.0->255
        # Row 1: min=100.0->0, max=300.0->255
        assert quantized[0, 0] == 0  # min of row 0
        assert quantized[0, 2] == 255  # max of row 0
        assert quantized[1, 0] == 0  # min of row 1
        assert quantized[1, 2] == 255  # max of row 1

    def test_quantization_with_negative_values(self):
        """Test all quantization types handle negative values correctly."""
        embedding = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        # int8: should preserve sign
        int8_result = SentenceTransformer._quantize_embedding(embedding, "int8")
        assert int8_result[0] < 0  # negative stays negative
        assert int8_result[4] > 0  # positive stays positive

        # uint8: should map to [0, 255] range
        uint8_result = SentenceTransformer._quantize_embedding(embedding, "uint8")
        assert np.all(uint8_result >= 0)

        # binary: should convert to -1/1
        binary_result = SentenceTransformer._quantize_embedding(embedding, "binary")
        assert binary_result[0] == -1
        assert binary_result[4] == 1

    def test_quantization_preserves_relative_ordering_int8(self):
        """Test int8 quantization preserves relative ordering of values."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        quantized = SentenceTransformer._quantize_embedding(embedding, "int8")

        # Monotonicity should be preserved
        assert np.all(quantized[:-1] <= quantized[1:])

    def test_quantization_edge_cases(self):
        """Test quantization handles edge cases."""
        # Empty-like arrays
        empty_1d = np.array([], dtype=np.float32)
        empty_2d = np.array([]).reshape(0, 0).astype(np.float32)

        for qtype in ["int8", "uint8", "binary"]:
            # Should handle without crashing
            result_1d = SentenceTransformer._quantize_embedding(empty_1d, qtype)
            result_2d = SentenceTransformer._quantize_embedding(empty_2d, qtype)
            assert result_1d.shape == empty_1d.shape
            assert result_2d.shape == empty_2d.shape
