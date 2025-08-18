# =============================================================================
# File: batch_limiter.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Iterator, List, TypeVar

from app.exceptions import BatchSizeError
from app.logger import get_logger

logger = get_logger("batch_limiter")

T = TypeVar("T")


class BatchLimiter:
    """Utility to limit batch sizes and prevent memory exhaustion."""

    @staticmethod
    def chunk_batch(items: List[T], max_batch_size: int = 10) -> Iterator[List[T]]:
        """Split large batch into smaller chunks."""
        if len(items) > max_batch_size:
            logger.warning(
                "Large batch size %d split into chunks of %d",
                len(items),
                max_batch_size,
            )

        for i in range(0, len(items), max_batch_size):
            yield items[i : i + max_batch_size]

    @staticmethod
    def validate_batch_size(items: List[T], max_size: int = 50) -> None:
        """Validate batch size doesn't exceed limits."""
        if len(items) > max_size:
            raise BatchSizeError(f"Batch size {len(items)} exceeds maximum {max_size}")
