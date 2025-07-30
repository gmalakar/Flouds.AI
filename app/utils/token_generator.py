# =============================================================================
# File: token_generator.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import secrets
import string
from typing import Optional


def generate_api_token(length: int = 32) -> str:
    """Generate a secure API token.

    Args:
        length: Token length (default: 32)

    Returns:
        Secure random token string
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_uuid_token() -> str:
    """Generate a UUID-based token."""
    import uuid

    return str(uuid.uuid4()).replace("-", "")


if __name__ == "__main__":
    # Generate tokens for testing
    print("Generated API Tokens:")
    print(f"32-char token: {generate_api_token()}")
    print(f"64-char token: {generate_api_token(64)}")
    print(f"UUID token: {generate_uuid_token()}")
