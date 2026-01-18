# =============================================================================
# File: __init__.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Embedder service for text embedding using ONNX models.

This package provides a modular embedder service with the following components:
- models: Data models and constants
- text_utils: Text preprocessing and manipulation
- onnx_utils: ONNX model input/output preparation
- processing: Embedding output processing, pooling, projection, quantization
- resource_manager: Resource loading (tokenizers, sessions, configs)
- inference: Core embedding inference logic
- embedder: Main SentenceTransformer class

Public API:
- SentenceTransformer: Main class for text embedding
- SingleEmbeddingResult: Result model for single embeddings
- ChunkEmbeddingResult: Result model for chunked embeddings
"""

from app.services.embedder.embedder import SentenceTransformer
from app.services.embedder.models import ChunkEmbeddingResult, SingleEmbeddingResult

__all__ = [
    "SentenceTransformer",
    "SingleEmbeddingResult",
    "ChunkEmbeddingResult",
]
