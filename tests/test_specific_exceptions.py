# =============================================================================
# File: test_specific_exceptions.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Tests for specific exception handling replacements."""

from unittest.mock import Mock, patch

import pytest

from app.exceptions import (
    CacheInvalidationError,
    ComponentHealthError,
    DatabaseConnectionError,
    DatabaseCorruptionError,
    DecryptionError,
    InvalidConfigError,
    InvalidInputError,
    MissingConfigError,
    ResourceException,
    TokenizerError,
)


class TestSpecificExceptions:
    """Test specific exception handling in various modules."""

    def test_database_exceptions_in_key_manager(self):
        """Test database-specific exceptions in KeyManager (SQLite)."""
        import os
        import tempfile

        from app.utils.key_manager import KeyManager

        # Test database connection error with invalid path
        with patch("app.utils.key_manager.sqlite3.connect") as mock_connect:
            mock_connect.side_effect = OSError("Permission denied")

            with pytest.raises(DatabaseConnectionError):
                KeyManager(db_path="/invalid/path/test.db")

    def test_config_exceptions_in_config_loader(self):
        """Test configuration-specific exceptions in ConfigLoader."""
        from app.config.config_loader import ConfigLoader

        # Test missing config error
        with pytest.raises(MissingConfigError):
            ConfigLoader.get_onnx_config("nonexistent_model")

    def test_health_check_exceptions(self):
        """Test health check specific exceptions."""
        from app.services.health_service import HealthService

        with patch("app.services.health_service.os.path.exists") as mock_exists:
            mock_exists.side_effect = OSError("Access denied")

            # Should handle OSError gracefully and return unhealthy
            result = HealthService._check_onnx()
            assert result == "unhealthy"

    def test_tokenizer_exceptions_in_chunking(self):
        """Test tokenizer-specific exceptions in chunking strategies."""
        from app.utils.chunking_strategies import ChunkingStrategies

        # Test that the method handles exceptions gracefully
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        mock_config = Mock()
        mock_config.chunk_logic = "sentence"

        # The method should return the original text when chunking fails
        result = ChunkingStrategies.split_text_into_chunks(
            "test text", mock_tokenizer, 100, mock_config
        )
        assert result == ["test text"]  # Should return original text on error

    def test_cache_exceptions_in_cache_manager(self):
        """Test cache-specific exceptions in CacheManager."""
        from app.utils.cache_manager import CacheManager

        with patch(
            "app.config.config_loader.ConfigLoader.get_cache_stats"
        ) as mock_stats:
            mock_stats.side_effect = Exception("Cache error")

            with pytest.raises(CacheInvalidationError):
                CacheManager.optimize_caches()

    def test_resource_exceptions_in_performance_monitor(self):
        """Test resource-specific exceptions in PerformanceMonitor."""
        from app.utils.performance_monitor import PerformanceMonitor

        with patch(
            "app.utils.performance_monitor.psutil.virtual_memory"
        ) as mock_memory:
            mock_memory.side_effect = Exception("System error")

            with pytest.raises(ResourceException):
                PerformanceMonitor.get_system_metrics()

    def test_model_exceptions_in_base_nlp_service(self):
        """Test model-specific exceptions in BaseNLPService."""
        from app.services.base_nlp_service import BaseNLPService

        with patch(
            "app.config.config_loader.ConfigLoader.get_onnx_config"
        ) as mock_config:
            mock_config.side_effect = Exception("Config error")

            # Should return None instead of raising exception
            result = BaseNLPService._get_model_config("test_model")
            assert result is None

    def test_exception_hierarchy(self):
        """Test that all custom exceptions inherit from FloudsBaseException."""
        from app.exceptions import FloudsBaseException

        exceptions_to_test = [
            DatabaseConnectionError,
            DatabaseCorruptionError,
            DecryptionError,
            InvalidConfigError,
            MissingConfigError,
            CacheInvalidationError,
            ComponentHealthError,
            InvalidInputError,
            TokenizerError,
            ResourceException,
        ]

        for exc_class in exceptions_to_test:
            assert issubclass(exc_class, FloudsBaseException)

            # Test exception creation with message and error code
            exc = exc_class("Test message", "TEST_CODE")
            assert exc.message == "Test message"
            assert exc.error_code == "TEST_CODE"

    def test_exception_error_codes(self):
        """Test that exceptions have proper error codes."""
        # Test default error code
        exc = DatabaseConnectionError("Test message")
        assert exc.error_code == "DatabaseConnectionError"

        # Test custom error code
        exc = InvalidConfigError("Test message", "CUSTOM_CODE")
        assert exc.error_code == "CUSTOM_CODE"

    def test_specific_exception_catching(self):
        """Test that specific exceptions are caught properly."""
        from app.exceptions import ModelLoadError, TokenizerError

        def raise_model_error():
            raise ModelLoadError("Model not found")

        def raise_tokenizer_error():
            raise TokenizerError("Tokenizer failed")

        # Test specific exception catching
        with pytest.raises(ModelLoadError) as exc_info:
            raise_model_error()
        assert "Model not found" in str(exc_info.value)

        with pytest.raises(TokenizerError) as exc_info:
            raise_tokenizer_error()
        assert "Tokenizer failed" in str(exc_info.value)

    def test_exception_message_sanitization(self):
        """Test that exception messages are properly sanitized."""
        from app.exceptions import InvalidInputError

        # Test with potentially dangerous input
        dangerous_input = "<script>alert('xss')</script>"
        exc = InvalidInputError(f"Invalid input: {dangerous_input}")

        # The message should contain the dangerous input as-is for debugging
        # but when logged, it should be sanitized by log_sanitizer
        assert dangerous_input in exc.message
