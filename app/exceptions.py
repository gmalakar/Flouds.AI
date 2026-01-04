# =============================================================================
# File: exceptions.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Custom exceptions for Flouds AI application."""
from typing import Optional


class FloudsBaseException(Exception):
    """Base exception for all Flouds AI errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        super().__init__(self.message)


class ModelException(FloudsBaseException):
    """Exceptions related to model loading and inference."""

    pass


class ModelNotFoundError(ModelException):
    """Model file or configuration not found."""

    pass


class ModelLoadError(ModelException):
    """Failed to load model or create session."""

    pass


class TokenizerError(ModelException):
    """Tokenizer-related errors."""

    pass


class InferenceError(ModelException):
    """Model inference failed."""

    pass


class ConfigurationException(FloudsBaseException):
    """Configuration-related errors."""

    pass


class InvalidConfigError(ConfigurationException):
    """Invalid configuration parameters."""

    pass


class MissingConfigError(ConfigurationException):
    """Required configuration missing."""

    pass


class ValidationException(FloudsBaseException):
    """Input validation errors."""

    pass


class InvalidInputError(ValidationException):
    """Invalid input parameters."""

    pass


class BatchSizeError(ValidationException):
    """Batch size exceeds limits."""

    pass


class TimeoutException(FloudsBaseException):
    """Operation timeout errors."""

    pass


class ProcessingTimeoutError(TimeoutException):
    """Processing operation timed out."""

    pass


class AuthenticationException(FloudsBaseException):
    """Authentication-related errors."""

    pass


class InvalidTokenError(AuthenticationException):
    """Invalid authentication token."""

    pass


class UnauthorizedError(AuthenticationException):
    """Unauthorized access attempt."""

    pass


class RateLimitException(FloudsBaseException):
    """Rate limiting errors."""

    pass


class RateLimitExceededError(RateLimitException):
    """Rate limit exceeded."""

    pass


class ResourceException(FloudsBaseException):
    """Resource-related errors."""

    pass


class InsufficientMemoryError(ResourceException):
    """Insufficient memory for operation."""

    pass


class DiskSpaceError(ResourceException):
    """Insufficient disk space."""

    pass


class DatabaseException(FloudsBaseException):
    """Database-related errors."""

    pass


class DatabaseConnectionError(DatabaseException):
    """Database connection failed."""

    pass


class DatabaseCorruptionError(DatabaseException):
    """Database file is corrupted."""

    pass


class EncryptionException(FloudsBaseException):
    """Encryption/decryption errors."""

    pass


class EncryptionKeyError(EncryptionException):
    """Invalid or missing encryption key."""

    pass


class DecryptionError(EncryptionException):
    """Failed to decrypt data."""

    pass


class CacheException(FloudsBaseException):
    """Cache-related errors."""

    pass


class CacheInvalidationError(CacheException):
    """Failed to invalidate cache."""

    pass


class HealthCheckException(FloudsBaseException):
    """Health check failures."""

    pass


class ComponentHealthError(HealthCheckException):
    """Component health check failed."""

    pass
