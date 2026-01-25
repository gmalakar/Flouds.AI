# =============================================================================
# File: text_utils.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Text processing and preprocessing utilities for prompt service."""

import re
from typing import Optional, Set

from app.logger import get_logger
from app.models.prompt_request import PromptRequest

logger = get_logger(__name__)


def prepare_input_text(
    request: PromptRequest, prepend_text: Optional[str] = None
) -> str:
    """Prepare input text for summarization by optionally prepending text.

    Args:
        request: The prompt request with input text
        prepend_text: Optional text to prepend to the input

    Returns:
        Prepared input text
    """
    input_text = _prepend_text(request.input, prepend_text)
    logger.debug(
        "Input text for summarization: %s",
        str(input_text[:100]) if input_text else "",
    )
    return input_text


def _prepend_text(text: str, prepend: Optional[str] = None) -> str:
    """Prepend optional text to the input.

    Args:
        text: The main input text
        prepend: Optional text to prepend

    Returns:
        Combined text (prepend + text)
    """
    if not prepend:
        return text
    return f"{prepend} {text}" if text else prepend


def remove_special_tokens(text: str, special_tokens: Set[str]) -> str:
    """Remove special tokens from text using regex for batch removal.

    Args:
        text: Input text
        special_tokens: Set of special tokens to remove

    Returns:
        Text with special tokens removed
    """
    if not special_tokens or not text:
        return text

    # Escape special regex characters and create pattern
    escaped_tokens = [re.escape(token) for token in special_tokens if token]
    if not escaped_tokens:
        return text

    pattern = "|".join(escaped_tokens)
    original_text = text
    text = re.sub(pattern, " ", text)

    if text != original_text:
        logger.debug(f"Removed special tokens using pattern: {pattern}")

    # Clean up extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def capitalize_sentences(text: str) -> str:
    """Capitalize the first word of each sentence.

    Args:
        text: Input text

    Returns:
        Text with capitalized sentence beginnings
    """
    if not text:
        return text

    sentences = re.split(r"([.!?]\s*)", text)
    result = []

    for i, part in enumerate(sentences):
        if i % 2 == 0 and part.strip():
            part = part.strip()
            if part:
                part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
        result.append(part)

    return "".join(result).strip()
