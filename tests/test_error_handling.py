# =============================================================================
# File: test_error_handling.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Tests for improved error handling system."""

import pytest

from app.exceptions import (
    BatchSizeError,
    InferenceError,
    InvalidTokenError,
    ModelLoadError,
    ModelNotFoundError,
    ProcessingTimeoutError,
    RateLimitExceededError,
    TokenizerError,
    UnauthorizedError,
)
from app.utils.error_handler import ErrorHandler, handle_errors


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_base_exception_properties(self):
        """Test base exception has correct properties."""
        exc = ModelNotFoundError("Model not found", "MODEL_404")
        assert exc.message == "Model not found"
        assert exc.error_code == "MODEL_404"
        assert str(exc) == "Model not found"

    def test_exception_default_error_code(self):
        """Test exception uses class name as default error code."""
        exc = InferenceError("Inference failed")
        assert exc.error_code == "InferenceError"

    def test_all_exception_types(self):
        """Test all custom exception types can be instantiated."""
        exceptions = [
            ModelNotFoundError("test"),
            ModelLoadError("test"),
            TokenizerError("test"),
            InferenceError("test"),
            ProcessingTimeoutError("test"),
            InvalidTokenError("test"),
            UnauthorizedError("test"),
            RateLimitExceededError("test"),
            BatchSizeError("test"),
        ]

        for exc in exceptions:
            assert exc.message == "test"
            assert exc.error_code is not None


class TestErrorHandler:
    """Test error handler utilities."""

    def test_handle_exception_custom(self):
        """Test handling custom exceptions."""
        exc = ModelNotFoundError("Model not found")
        result = ErrorHandler.handle_exception(exc, "test_context")

        assert result["success"] is False
        assert result["message"] == "Model not found"
        assert result["error_code"] == "ModelNotFoundError"
        assert result["context"] == "test_context"

    def test_handle_exception_builtin(self):
        """Test handling built-in exceptions."""
        exc = FileNotFoundError("File not found")
        result = ErrorHandler.handle_exception(exc, "test_context")

        assert result["success"] is False
        assert result["message"] == "Model files not accessible"
        assert result["error_code"] == "FILE_NOT_FOUND"

    def test_get_http_status(self):
        """Test HTTP status code mapping."""
        assert ErrorHandler.get_http_status(InvalidTokenError("test")) == 401
        assert ErrorHandler.get_http_status(RateLimitExceededError("test")) == 429
        assert ErrorHandler.get_http_status(ModelLoadError("test")) == 503
        assert ErrorHandler.get_http_status(ProcessingTimeoutError("test")) == 504
        assert ErrorHandler.get_http_status(ValueError("test")) == 500


class TestErrorDecorators:
    """Test error handling decorators."""

    def test_handle_errors_decorator(self):
        """Test error handling decorator."""

        @handle_errors(context="test_function")
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "Invalid parameter value" in result["message"]


if __name__ == "__main__":
    pytest.main([__file__])
