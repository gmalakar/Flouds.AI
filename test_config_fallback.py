# =============================================================================
# File: test_config_fallback.py
# Date: 2025-08-18
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

#!/usr/bin/env python3
"""
Test script to verify that cached config values are used as fallback
when request parameters are None.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from unittest.mock import Mock

from app.services.embedder_service import SentenceTransformer


def test_config_fallback():
    """Test that cached config values are used when request params are None."""

    # Create a mock config with some values
    mock_config = Mock()
    mock_config.pooling_strategy = "max"  # Cached value
    mock_config.max_length = 512  # Cached value
    mock_config.normalize = False  # Cached value
    mock_config.chunk_logic = "paragraph"  # Cached value

    # Test case 1: Request param is None - should use cached config
    request_params = {
        "pooling_strategy": None,  # Should fallback to "max"
        "max_length": 256,  # Should override to 256
        "normalize": None,  # Should fallback to False
        "chunk_logic": "sentence",  # Should override to "sentence"
    }

    result_config = SentenceTransformer._override_config_with_request(
        mock_config, request_params
    )

    # Verify fallback behavior
    assert (
        getattr(result_config, "pooling_strategy") == "max"
    ), f"Expected 'max', got {getattr(result_config, 'pooling_strategy')}"
    assert (
        getattr(result_config, "max_length") == 256
    ), f"Expected 256, got {getattr(result_config, 'max_length')}"
    assert (
        getattr(result_config, "normalize") == False
    ), f"Expected False, got {getattr(result_config, 'normalize')}"
    assert (
        getattr(result_config, "chunk_logic") == "sentence"
    ), f"Expected 'sentence', got {getattr(result_config, 'chunk_logic')}"

    print("[PASS] Test 1 passed: None values correctly fall back to cached config")

    # Test case 2: All request params are None - should use all cached values
    request_params_all_none = {
        "pooling_strategy": None,
        "max_length": None,
        "normalize": None,
        "chunk_logic": None,
    }

    # Reset mock config
    mock_config.pooling_strategy = "cls"
    mock_config.max_length = 1024
    mock_config.normalize = True
    mock_config.chunk_logic = "fixed"

    result_config2 = SentenceTransformer._override_config_with_request(
        mock_config, request_params_all_none
    )

    assert getattr(result_config2, "pooling_strategy") == "cls"
    assert getattr(result_config2, "max_length") == 1024
    assert getattr(result_config2, "normalize") == True
    assert getattr(result_config2, "chunk_logic") == "fixed"

    print("[PASS] Test 2 passed: All None values correctly fall back to cached config")

    # Test case 3: No request params provided - should keep cached values
    mock_config.pooling_strategy = "mean"
    mock_config.max_length = 128

    result_config3 = SentenceTransformer._override_config_with_request(mock_config, {})

    assert getattr(result_config3, "pooling_strategy") == "mean"
    assert getattr(result_config3, "max_length") == 128

    print("[PASS] Test 3 passed: Empty request params keep cached config values")

    print(
        "\n[SUCCESS] All tests passed! Cached config values are correctly used as fallback."
    )


if __name__ == "__main__":
    test_config_fallback()
