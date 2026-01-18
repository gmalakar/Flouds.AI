# =============================================================================
# File: embedder_service.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""
Backward compatibility wrapper for embedder service.

This file maintains the original import path while delegating to the refactored
modular embedder package.

The embedder service has been refactored into the following modules:
- app/services/embedder/models.py: Data models and constants
- app/services/embedder/text_utils.py: Text preprocessing
- app/services/embedder/onnx_utils.py: ONNX input/output handling
- app/services/embedder/processing.py: Embedding processing and projection
- app/services/embedder/resource_manager.py: Resource loading
- app/services/embedder/inference.py: Core inference logic
- app/services/embedder/embedder.py: Main SentenceTransformer class

All public APIs remain unchanged for backward compatibility.
"""

# Import all public symbols from refactored embedder package
from app.services.embedder import ChunkEmbeddingResult, SentenceTransformer, SingleEmbeddingResult

# Re-export validate_safe_path for backward compatibility with tests
from app.utils.path_validator import validate_safe_path

# Re-export for backward compatibility
__all__ = [
    "SentenceTransformer",
    "SingleEmbeddingResult",
    "validate_safe_path",
    "ChunkEmbeddingResult",
]
