# =============================================================================
# File: pooling_strategies.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import numpy as np

from app.logger import get_logger

logger = get_logger("pooling_strategies")


class PoolingStrategies:

    @staticmethod
    def should_skip_pooling(
        embedding, attention_mask, force_pooling=False, strategy="cls"
    ):
        if force_pooling:
            return False

        # Skip pooling for single embeddings (1, embedding_dim) -> (embedding_dim,)
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            return True

        # Check for single-token input (excluding padding)
        if attention_mask is not None:
            num_active_tokens = attention_mask.sum()
            if num_active_tokens == 1 and strategy in ["cls", "first"]:
                return True

        return False

    @staticmethod
    def mean_pooling(embedding: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean pooling with attention mask."""
        assert (
            embedding.shape[:2] == attention_mask.shape
        ), "Embedding and attention mask dimensions mismatch"
        masked_embedding = embedding * attention_mask[..., None]
        sum_embedding = masked_embedding.sum(axis=1)
        sum_mask = attention_mask.sum(axis=1, keepdims=True)
        return sum_embedding / np.maximum(sum_mask, 1e-9)

    @staticmethod
    def max_pooling(embedding: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Max pooling with attention mask."""
        assert (
            embedding.shape[:2] == attention_mask.shape
        ), "Embedding and attention mask dimensions mismatch"
        # Set masked positions to large negative value before max
        masked_embedding = np.where(
            attention_mask[..., None].astype(bool), embedding, -1e9
        )
        return masked_embedding.max(axis=1)

    @staticmethod
    def apply(
        embedding: np.ndarray,
        strategy: str = "mean",
        attention_mask: np.ndarray = None,
        force_pooling: bool = False,
    ) -> np.ndarray:

        logger.debug(f"Applying pooling strategy: {strategy}")

        if embedding.ndim == 1:
            logger.debug("Pooling has skipped 1D embedding.")
            return embedding

        """Apply pooling strategy to embeddings with attention mask support."""
        if PoolingStrategies.should_skip_pooling(
            embedding, attention_mask, force_pooling, strategy
        ):
            logger.debug("Pooling has skipped 1D embedding 2.")
            return embedding[0]

        if strategy in ["cls", "first"]:
            return embedding[0] if embedding.ndim >= 2 else embedding

        elif strategy == "last":
            if attention_mask is not None and embedding.ndim == 3:
                # Get last non-padded token for each sequence
                indices = np.maximum(attention_mask.sum(axis=1) - 1, 0)
                return embedding[np.arange(embedding.shape[0]), indices]
            return embedding[-1] if embedding.ndim >= 2 else embedding

        elif strategy == "max":
            if attention_mask is not None and embedding.ndim == 3:
                return PoolingStrategies.max_pooling(embedding, attention_mask)
            return embedding.max(axis=1) if embedding.ndim >= 2 else embedding

        # Default to mean pooling
        if attention_mask is not None and embedding.ndim == 3:
            logger.debug(f"Applying pooling strategy: {strategy} has applied")
            return PoolingStrategies.mean_pooling(embedding, attention_mask)

        logger.debug(
            f"Applying pooling strategy without attention mask: {strategy} has applied"
        )
        return embedding.mean(axis=1) if embedding.ndim >= 2 else embedding
