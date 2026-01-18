# =============================================================================
# File: text_utils.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Text preprocessing and manipulation utilities for embeddings."""

import re
import unicodedata
from typing import Any, List

import numpy as np

from app.models.embedded_chunk import EmbededChunk


def preprocess_text(text: str, lowercase: bool = True, remove_emojis: bool = False) -> str:
    """Clean and normalize raw text for embedding.

    Args:
        text: Raw input text
        lowercase: Whether to convert text to lowercase
        remove_emojis: Whether to remove emoji and non-ASCII characters

    Returns:
        Preprocessed and normalized text
    """
    # Normalize Unicode characters (e.g. curly quotes, accented letters)
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML or XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Replace all types of whitespace (tabs, line breaks, multiple spaces) with single space
    text = re.sub(r"\s+", " ", text)

    # Optional: Remove emojis and non-ASCII characters
    if remove_emojis:
        text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Optional: Convert to lowercase
    if lowercase:
        text = text.lower()

    # Final cleanup: trim leading and trailing whitespace
    text = text.strip()

    return text


def prepare_text_for_embedding(text: str, model_config: Any) -> str:
    """Prepare text for embedding processing using model config settings.

    Args:
        text: Raw input text
        model_config: Model configuration with preprocessing settings

    Returns:
        Text ready for tokenization and embedding
    """
    lowercase = getattr(model_config, "lowercase", True)
    remove_emojis = getattr(model_config, "remove_emojis", False)
    return preprocess_text(text, lowercase, remove_emojis)


def merge_vectors(chunks: List[EmbededChunk], method: str = "mean") -> List[float]:
    """Merge embedding vectors using pooling strategy.

    Args:
        chunks: List of embedded chunks with vectors
        method: Pooling method - "mean" or "max"

    Returns:
        Merged vector as list of floats
    """
    vectors = [np.array(chunk.vector) for chunk in chunks if hasattr(chunk, "vector")]
    if not vectors:
        return []

    # Validate pooling method
    if method not in ["mean", "max"]:
        method = "mean"

    stacked = np.stack(vectors)
    merged = np.max(stacked, axis=0) if method == "max" else np.mean(stacked, axis=0)
    return merged.tolist()
