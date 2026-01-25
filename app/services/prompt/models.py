# =============================================================================
# File: models.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Data models and constants for prompt processing service."""

import os

# typing imports not required in this module
from pydantic import BaseModel, Field

from app.modules.concurrent_dict import ConcurrentDict

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL = "t5-small"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 20
DEFAULT_VOCAB_SIZE = 32000
MEMORY_LOW_THRESHOLD_MB = 150  # Clear cache if available memory below this

# Cache size limits (from environment or defaults)
CACHE_LIMIT_DECODER = int(os.getenv("FLOUDS_DECODER_CACHE_MAX", "3"))
CACHE_LIMIT_ENCODER = int(os.getenv("FLOUDS_ENCODER_CACHE_MAX", "3"))
CACHE_LIMIT_MODELS = int(os.getenv("FLOUDS_MODEL_CACHE_MAX", "2"))
CACHE_LIMIT_SPECIAL = int(os.getenv("FLOUDS_SPECIAL_TOKENS_CACHE_MAX", "8"))


# ============================================================================
# Pydantic Models
# ============================================================================


class SummaryResults(BaseModel):
    """Result model for text summarization/generation."""

    summary: str
    message: str
    success: bool = Field(default=True)


# ============================================================================
# Cached Collections (Class-level shared state)
# ============================================================================


class CachedSessions:
    """Manages cached ONNX sessions and models for PromptProcessor."""

    decoder_sessions: ConcurrentDict = ConcurrentDict(
        "_decoder_sessions", max_size=CACHE_LIMIT_DECODER
    )
    encoder_sessions: ConcurrentDict = ConcurrentDict(
        "_encoder_sessions", max_size=CACHE_LIMIT_ENCODER
    )
    models: ConcurrentDict = ConcurrentDict("_models", max_size=CACHE_LIMIT_MODELS)
    special_tokens: ConcurrentDict = ConcurrentDict("_special_tokens", max_size=CACHE_LIMIT_SPECIAL)

    @classmethod
    def clear_all(cls) -> None:
        """Clear all cached sessions and models."""
        cls.decoder_sessions.clear()
        cls.encoder_sessions.clear()
        cls.models.clear()
        cls.special_tokens.clear()
