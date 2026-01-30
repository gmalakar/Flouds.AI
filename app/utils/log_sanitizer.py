# =============================================================================
# File: log_sanitizer.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import json
import re
from typing import Any


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


def sanitize_body(body: bytes, max_bytes: int = 200) -> str:
    """Create a safe, redacted snippet from request/response body bytes.

    - If the body is JSON, redact common sensitive keys before returning
      a truncated JSON snippet.
    - Otherwise, return a truncated UTF-8 decoded snippet or a binary
      placeholder.
    """
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            sensitive_keys = {"password", "token", "secret", "api_key", "apikey", "authorization"}
            sanitized = {
                k: "***REDACTED***" if k.lower() in sensitive_keys else v for k, v in parsed.items()
            }
            dumped = json.dumps(sanitized).encode("utf-8", errors="replace")
            snippet = dumped[:max_bytes].decode("utf-8", errors="replace")
            return snippet + "..." if len(dumped) >= max_bytes else snippet
        dumped = json.dumps(parsed).encode("utf-8", errors="replace")
        snippet = dumped[:max_bytes].decode("utf-8", errors="replace")
        return snippet + "..." if len(dumped) >= max_bytes else snippet
    except Exception:
        try:
            snippet = body[:max_bytes].decode("utf-8", errors="replace")
            return snippet + "..." if len(body) >= max_bytes else snippet
        except Exception:
            return f"<binary {len(body)} bytes>"
