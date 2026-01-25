# =============================================================================
# File: __init__.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Prompt service module - Text generation and summarization.

This module provides a refactored prompt processing service decomposed
into focused, maintainable components:

- processor: Main orchestrator (PromptProcessor class)
- generator: Generation strategies (seq2seq, encoder-only, ONNX)
- config: Model configuration resolution
- resource_manager: Model/session caching and lifecycle
- text_utils: Text preprocessing and postprocessing
- parameters: Generation parameter building
- models: Data models and constants
"""

from app.services.prompt.models import SummaryResults

__all__ = ["SummaryResults"]
