# =============================================================================
# File: log_sanitizer.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import re
from typing import Any, Union


def sanitize_for_log(value: Any) -> str:
    """
    Sanitize input for safe logging by removing/encoding dangerous characters.

    Args:
        value: Input value to sanitize

    Returns:
        str: Sanitized string safe for logging
    """
    if value is None:
        return "None"

    # Convert to string
    str_value = str(value)

    # Remove or replace dangerous characters
    # Remove newlines, carriage returns, and other control characters
    sanitized = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", "_", str_value)

    # Limit length to prevent log flooding
    if len(sanitized) > 200:
        sanitized = sanitized[:197] + "..."

    return sanitized


def sanitize_dict_for_log(data: dict) -> dict:
    """
    Sanitize dictionary values for logging.

    Args:
        data: Dictionary to sanitize

    Returns:
        dict: Dictionary with sanitized values
    """
    return {k: sanitize_for_log(v) for k, v in data.items()}
