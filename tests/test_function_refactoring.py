# =============================================================================
# File: test_function_refactoring.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Tests to verify refactored functions work correctly."""

from unittest.mock import Mock, patch

import pytest

from app.services.embedder_service import SentenceTransformer
from app.services.prompt_service import PromptProcessor


class TestSummarizerRefactoring:
    """Test refactored summarizer methods."""

    def test_prepare_model_resources(self):
        """Test model resource preparation."""
        with patch(
            "app.services.base_nlp_service.BaseNLPService._get_model_config"
        ) as mock_config, patch.object(
            PromptProcessor, "_get_tokenizer_threadsafe"
        ) as mock_tokenizer, patch(
            "app.services.base_nlp_service.BaseNLPService._get_model_path"
        ) as mock_model_path:

            mock_config.return_value = Mock(
                tasks=["summarization"],
                legacy_tokenizer=False,
                model_folder_name="test-model",
            )
            mock_tokenizer.return_value = Mock()
            mock_model_path.return_value = "/safe/path"

            model_path, tokenizer = PromptProcessor._prepare_model_resources(
                mock_config.return_value, "test-model"
            )

            assert model_path == "/safe/path"
            assert tokenizer is not None
            mock_tokenizer.assert_called_once()

    def test_build_generation_params(self):
        """Test generation parameter building."""
        mock_config = Mock(
            tasks=["summarization"],
            legacy_tokenizer=False,
            model_folder_name="test-model",
        )
        mock_config.max_length = 256
        mock_config.min_length = 10
        mock_config.num_beams = 4
        mock_config.early_stopping = True
        mock_config.temperature = 0.7

        mock_request = Mock()
        mock_request.temperature = 0.8

        params = PromptProcessor._build_generation_params(mock_config, mock_request)

        assert params["max_length"] == 256
        assert params["min_length"] == 10
        assert params["num_beams"] == 4
        assert params["early_stopping"] is True
        assert params["temperature"] == 0.8
        assert params["do_sample"] is True

    def test_get_model_file_paths(self):
        """Test model file path generation."""
        with patch("app.services.prompt_service.validate_safe_path") as mock_path:
            mock_path.side_effect = lambda x, y: x  # Return path as-is

            mock_config = Mock()
            mock_config.use_optimized = True
            mock_config.encoder_optimized_onnx_model = "encoder_opt.onnx"
            mock_config.decoder_optimized_onnx_model = "decoder_opt.onnx"

            encoder_path, decoder_path = PromptProcessor._get_model_file_paths(
                "/model/path", mock_config
            )

            assert "encoder_opt.onnx" in encoder_path
            assert "decoder_opt.onnx" in decoder_path


class TestEmbedderRefactoring:
    """Test refactored embedder methods."""

    def test_prepare_text_for_embedding(self):
        """Test text preparation for embedding."""
        mock_config = Mock()
        mock_config.lowercase = True
        mock_config.remove_emojis = False

        with patch.object(SentenceTransformer, "_preprocess_text") as mock_preprocess:
            mock_preprocess.return_value = "processed text"

            result = SentenceTransformer._prepare_text_for_embedding(
                "Test Text", mock_config
            )

            assert result == "processed text"
            mock_preprocess.assert_called_once_with("Test Text", True, False)

    def test_prepare_embedding_resources(self):
        """Test embedding resource preparation."""
        with patch.object(
            SentenceTransformer, "_get_model_config"
        ) as mock_config, patch.object(
            SentenceTransformer, "_get_model_path"
        ) as mock_model_path, patch.object(
            SentenceTransformer, "_get_tokenizer_threadsafe"
        ) as mock_tokenizer, patch.object(
            SentenceTransformer, "_get_encoder_session"
        ) as mock_session, patch.object(
            SentenceTransformer, "_get_embedding_model_path"
        ) as mock_embedding_path:

            mock_config_obj = Mock()
            mock_config_obj.tasks = ["embedding"]
            mock_config_obj.model_folder_name = "test-model"
            mock_config_obj.legacy_tokenizer = False
            mock_config.return_value = mock_config_obj
            mock_model_path.return_value = "/safe/model/path"
            mock_tokenizer.return_value = Mock()
            mock_session.return_value = Mock()
            mock_embedding_path.return_value = "/safe/embedding/path"

            config, tokenizer, session = (
                SentenceTransformer._prepare_embedding_resources("test-model")
            )

            assert config is not None
            assert tokenizer is not None
            assert session is not None

    def test_get_embedding_model_path(self):
        """Test embedding model path generation."""
        with patch("app.services.embedder_service.validate_safe_path") as mock_path:
            mock_path.return_value = "/safe/model/path"

            mock_config = Mock()
            mock_config.use_optimized = False
            mock_config.encoder_onnx_model = "custom_model.onnx"

            result = SentenceTransformer._get_embedding_model_path(
                "/base/path", mock_config
            )

            assert result == "/safe/model/path"
            mock_path.assert_called_once()

    def test_add_optional_inputs(self):
        """Test optional input addition with auto-detection."""
        inputs = {"input_ids": Mock()}
        mock_input_names = Mock()
        mock_input_names.position = "position_ids"
        mock_input_names.tokentype = "token_type_ids"
        mock_input_names.use_decoder_input = False

        # Create mock inputs that include both position_ids and token_type_ids
        mock_input1 = Mock()
        mock_input1.name = "input_ids"
        mock_input2 = Mock()
        mock_input2.name = "token_type_ids"
        mock_input3 = Mock()
        mock_input3.name = "position_ids"
        mock_session = Mock()
        mock_session.get_inputs.return_value = [mock_input1, mock_input2, mock_input3]

        SentenceTransformer._add_optional_inputs(
            inputs, mock_input_names, Mock(), mock_session, 128
        )

        # Both should be added since they're in model inputs
        assert "position_ids" in inputs
        assert "token_type_ids" in inputs


class TestFunctionComplexity:
    """Test that refactored functions are simpler."""

    def test_function_line_counts(self):
        """Verify functions are reasonably sized."""
        import inspect

        # Check summarizer functions
        summarizer_methods = [
            PromptProcessor._prepare_model_resources,
            PromptProcessor._build_generation_params,
            PromptProcessor._get_model_file_paths,
            PromptProcessor._get_token_config,
            PromptProcessor._sample_next_token,
        ]

        for method in summarizer_methods:
            lines = len(inspect.getsource(method).split("\n"))
            # Allow more lines for _sample_next_token due to advanced sampling logic
            max_lines = 70 if method.__name__ == "_sample_next_token" else 50
            assert (
                lines < max_lines
            ), f"{method.__name__} has {lines} lines, should be < {max_lines}"

        # Check embedder functions
        embedder_methods = [
            SentenceTransformer._prepare_text_for_embedding,
            SentenceTransformer._prepare_embedding_resources,
            SentenceTransformer._get_embedding_model_path,
            SentenceTransformer._add_optional_inputs,
            SentenceTransformer._process_text_chunks,
        ]

        for method in embedder_methods:
            lines = len(inspect.getsource(method).split("\n"))
            assert lines < 50, f"{method.__name__} has {lines} lines, should be < 50"


if __name__ == "__main__":
    pytest.main([__file__])
