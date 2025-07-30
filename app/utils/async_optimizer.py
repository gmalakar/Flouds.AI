# =============================================================================
# File: async_optimizer.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, TypeVar

from app.logger import get_logger

logger = get_logger("async_optimizer")

T = TypeVar("T")


class AsyncOptimizer:
    """Optimized async processing with connection pooling and task management."""

    _executor_pool: ThreadPoolExecutor = None
    _max_workers: int = 4

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if cls._executor_pool is None:
            cls._executor_pool = ThreadPoolExecutor(max_workers=cls._max_workers)
            logger.info(f"Created thread pool with {cls._max_workers} workers")
        return cls._executor_pool

    @classmethod
    async def process_batch_optimized(
        cls, items: List[T], process_func: Callable[[T], Any], max_concurrent: int = 10
    ) -> List[Any]:
        """Process batch with concurrency limits and optimized task management."""

        if not items:
            return []

        # Use semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        executor = cls.get_executor()

        async def process_item(item: T) -> Any:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(executor, process_func, item)

        # Process all items concurrently with limits
        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug(
            f"Processed {len(items)} items with {max_concurrent} max concurrent"
        )
        return results

    @classmethod
    def cleanup(cls) -> None:
        """Cleanup thread pool resources."""
        if cls._executor_pool:
            cls._executor_pool.shutdown(wait=True)
            cls._executor_pool = None
            logger.info("Thread pool executor cleaned up")
