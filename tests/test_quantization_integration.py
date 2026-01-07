# =============================================================================
# File: test_quantization_integration.py
# Date: 2026-01-06
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""
Integration tests for quantization with full embedding pipeline.
Tests model config defaults and request-level overrides.
"""

import numpy as np
import pytest

from app.config.onnx_config import OnnxConfig
from app.models.embedding_request import EmbeddingBatchRequest, EmbeddingRequest
from app.services.embedder_service import SentenceTransformer


class TestQuantizationIntegration:
    """Integration tests for quantization in embedding pipeline."""

    @pytest.fixture
    def mock_model_config(self):
        """Create mock model config with quantization enabled."""
        config = OnnxConfig(
            dimension=128,
            max_length=256,
            normalize=True,
            quantize=True,
            quantize_type="int8",
        )
        return config

    @pytest.fixture
    def mock_model_config_no_quantize(self):
        """Create mock model config with quantization disabled."""
        config = OnnxConfig(
            dimension=128, max_length=256, normalize=True, quantize=False
        )
        return config

    def test_quantization_applied_with_model_default(self, mock_model_config):
        """Test quantization is applied when model config has it enabled."""
        embedding = np.array([0.5, -0.3, 0.8, 0.0], dtype=np.float32)

        # Process with quantization enabled in model config
        result = SentenceTransformer._process_embedding_output(
            embedding, mock_model_config, None, None
        )

        # Should be quantized to int8
        assert result.dtype == np.int8
        assert np.all(result >= -127)
        assert np.all(result <= 127)

    def test_quantization_skipped_with_model_default(
        self, mock_model_config_no_quantize
    ):
        """Test quantization is skipped when model config has it disabled."""
        embedding = np.array([0.5, -0.3, 0.8, 0.0], dtype=np.float32)

        # Process with quantization disabled in model config
        result = SentenceTransformer._process_embedding_output(
            embedding, mock_model_config_no_quantize, None, None
        )

        # Should remain float32
        assert result.dtype == np.float32

    def test_request_override_enables_quantization(self, mock_model_config_no_quantize):
        """Test request-level parameter overrides model config to enable quantization."""
        embedding = np.array([0.5, -0.3, 0.8, 0.0], dtype=np.float32)

        # Model has quantization disabled, but request enables it
        result = SentenceTransformer._process_embedding_output(
            embedding,
            mock_model_config_no_quantize,
            None,
            None,
            quantize=True,
            quantize_type="int8",
        )

        # Should be quantized
        assert result.dtype == np.int8

    def test_request_override_disables_quantization(self, mock_model_config):
        """Test request-level parameter overrides model config to disable quantization."""
        embedding = np.array([0.5, -0.3, 0.8, 0.0], dtype=np.float32)

        # Model has quantization enabled, but request disables it
        result = SentenceTransformer._process_embedding_output(
            embedding, mock_model_config, None, None, quantize=False
        )

        # Should remain float32
        assert result.dtype == np.float32

    def test_request_override_quantization_type(self, mock_model_config):
        """Test request-level parameter overrides quantization type."""
        embedding = np.array([0.5, -0.3, 0.8, 0.0], dtype=np.float32)

        # Model has int8, but request wants binary
        result = SentenceTransformer._process_embedding_output(
            embedding,
            mock_model_config,
            None,
            None,
            quantize=True,
            quantize_type="binary",
        )

        # Should be binary quantized (-1 or 1)
        assert result.dtype == np.int8
        assert np.all(np.isin(result, [-1, 1]))

    def test_quantization_after_normalization(self, mock_model_config):
        """Test that quantization is applied AFTER normalization."""
        # Non-normalized embedding
        embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        result = SentenceTransformer._process_embedding_output(
            embedding, mock_model_config, None, None, normalize=True, quantize=True
        )

        # Should be normalized then quantized
        assert result.dtype == np.int8
        # Verify normalization happened (values should be distributed, not all clipped at 127)
        assert result.max() < 127

    def test_quantization_after_projection(self):
        """Test that quantization is applied AFTER projection."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        config = OnnxConfig(
            dimension=128,
            max_length=256,
            normalize=True,
            quantize=True,
            quantize_type="int8",
        )

        # Project to smaller dimension
        result = SentenceTransformer._process_embedding_output(
            embedding, config, None, projected_dimension=2
        )

        # Should be projected (dimension changed) and quantized
        assert result.dtype == np.int8
        assert result.shape[0] == 2  # Projected dimension

    def test_batch_quantization(self, mock_model_config):
        """Test quantization works correctly for batch embeddings."""
        embeddings = np.array(
            [[0.5, -0.5, 0.0], [1.0, 0.0, -1.0], [-0.25, 0.75, 0.1]], dtype=np.float32
        )
        # Disable pooling to preserve batch shape
        mock_model_config.pooling_strategy = "none"

        result = SentenceTransformer._process_embedding_output(
            embeddings, mock_model_config, None, None
        )

        # Should be batch quantized
        assert result.dtype == np.int8
        assert result.shape == embeddings.shape

    def test_uint8_quantization_integration(self):
        """Test uint8 quantization in full pipeline."""
        embedding = np.array([1.5, 2.0, 1.0, 1.75, 1.25], dtype=np.float32)
        config = OnnxConfig(
            dimension=128,
            max_length=256,
            normalize=False,  # Don't normalize for uint8 test
            quantize=True,
            quantize_type="uint8",
        )

        result = SentenceTransformer._process_embedding_output(
            embedding, config, None, None
        )

        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_binary_quantization_integration(self):
        """Test binary quantization in full pipeline."""
        embedding = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)
        config = OnnxConfig(
            dimension=128,
            max_length=256,
            normalize=True,
            quantize=True,
            quantize_type="binary",
        )

        result = SentenceTransformer._process_embedding_output(
            embedding, config, None, None
        )

        assert result.dtype == np.int8
        assert np.all(np.isin(result, [-1, 1]))

    def test_embedding_request_model_has_quantize_fields(self):
        """Test EmbeddingRequest model accepts quantization parameters."""
        request = EmbeddingRequest(
            model="test-model",
            input="test text",
            quantize=True,
            quantize_type="int8",
        )

        assert request.quantize is True
        assert request.quantize_type == "int8"

    def test_embedding_batch_request_model_has_quantize_fields(self):
        """Test EmbeddingBatchRequest model accepts quantization parameters."""
        request = EmbeddingBatchRequest(
            model="test-model",
            inputs=["text1", "text2"],
            quantize=True,
            quantize_type="binary",
        )

        assert request.quantize is True
        assert request.quantize_type == "binary"

    def test_quantization_with_all_pipeline_features(self):
        """Test quantization works with normalization and projection."""
        embedding = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        config = OnnxConfig(
            dimension=128,
            max_length=256,
            normalize=True,
            quantize=True,
            quantize_type="int8",
            pooling_strategy="none",  # Disable pooling to test batch projection
        )

        result = SentenceTransformer._process_embedding_output(
            embedding,
            config,
            None,
            projected_dimension=2,
            normalize=True,
            quantize=True,
        )

        # Should be projected, normalized, and quantized
        assert result.dtype == np.int8
        assert result.shape == (2, 2)  # Batch size 2, projected dim 2

    def test_onnx_config_has_quantization_fields(self):
        """Test OnnxConfig model has quantization fields with correct defaults."""
        config = OnnxConfig()

        assert hasattr(config, "quantize")
        assert hasattr(config, "quantize_type")
        assert config.quantize is False  # Default disabled
        assert config.quantize_type == "int8"  # Default type

    def test_onnx_config_quantization_fields_validation(self):
        """Test OnnxConfig validates quantization fields."""
        # Should accept valid values
        config = OnnxConfig(quantize=True, quantize_type="uint8")
        assert config.quantize is True
        assert config.quantize_type == "uint8"

        # Should accept all quantization types
        for qtype in ["int8", "uint8", "binary"]:
            config = OnnxConfig(quantize=True, quantize_type=qtype)
            assert config.quantize_type == qtype

    def test_storage_reduction_achieved(self):
        """Test that quantization actually reduces storage in memory."""
        embedding = np.random.randn(1536).astype(np.float32)
        config = OnnxConfig(
            dimension=1536,
            normalize=True,
            quantize=True,
            quantize_type="int8",
        )

        float_result = SentenceTransformer._process_embedding_output(
            embedding.copy(), config, None, None, quantize=False
        )
        quantized_result = SentenceTransformer._process_embedding_output(
            embedding.copy(), config, None, None, quantize=True
        )

        # Quantized should be 4x smaller
        assert quantized_result.nbytes == float_result.nbytes // 4

    def test_quantization_preserves_cosine_similarity_ranking(self):
        """Test that quantization preserves relative similarity ranking."""
        # Three embeddings with known similarities
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)  # Very similar to emb1
        emb3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Orthogonal to emb1

        config = OnnxConfig(normalize=True, quantize=True, quantize_type="int8")

        q1 = SentenceTransformer._process_embedding_output(
            emb1, config, None, None
        ).astype(np.float32)
        q2 = SentenceTransformer._process_embedding_output(
            emb2, config, None, None
        ).astype(np.float32)
        q3 = SentenceTransformer._process_embedding_output(
            emb3, config, None, None
        ).astype(np.float32)

        # Compute cosine similarities
        sim_1_2 = np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))
        sim_1_3 = np.dot(q1, q3) / (np.linalg.norm(q1) * np.linalg.norm(q3))

        # Similarity ranking should be preserved
        assert sim_1_2 > sim_1_3, "Quantization should preserve similarity ranking"
