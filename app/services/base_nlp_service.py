# =============================================================================
# File: base_nlp_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import threading
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizer

from app.app_init import APP_SETTINGS
from app.config.config_loader import ConfigLoader
from app.logger import get_logger
from app.modules.concurrent_dict import ConcurrentDict

logger = get_logger("base_nlp_service")

# Thread-local storage for tokenizers
_tokenizer_local = threading.local()


class BaseNLPService:
    _root_path: str = APP_SETTINGS.onnx.onnx_path
    _encoder_sessions: ConcurrentDict = ConcurrentDict("_encoder_sessions")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = np.asarray(x)
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def _get_model_config(model_to_use: str) -> Any:
        """Load ONNX model configuration."""
        return ConfigLoader.get_onnx_config(model_to_use)

    @staticmethod
    def _get_tokenizer_threadsafe(
        tokenizer_path: str, use_legacy: bool = False
    ) -> Optional[PreTrainedTokenizer]:
        """Thread-safe tokenizer loading with error handling."""
        try:
            if not hasattr(_tokenizer_local, "tokenizers"):
                _tokenizer_local.tokenizers = {}

            cache_key = f"{tokenizer_path}#{use_legacy}"
            if cache_key not in _tokenizer_local.tokenizers:
                if os.path.exists(tokenizer_path):
                    if use_legacy:
                        # Use legacy tokenizer for older models
                        tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_path, local_files_only=True, legacy=True
                        )
                    else:
                        try:
                            # Try loading with current transformers version
                            tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_path, local_files_only=True
                            )
                        except Exception:
                            # Fallback: try with legacy=True for older tokenizers
                            tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_path, local_files_only=True, legacy=True
                            )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

                _tokenizer_local.tokenizers[cache_key] = tokenizer

            return _tokenizer_local.tokenizers[cache_key]

        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            return None

    @staticmethod
    def _get_encoder_session(encoder_model_path: str) -> Optional[ort.InferenceSession]:
        """Get or create ONNX session with error handling."""
        try:
            provider = APP_SETTINGS.server.session_provider or "CPUExecutionProvider"
            cache_key = f"{encoder_model_path}#{provider}"

            return BaseNLPService._encoder_sessions.get_or_add(
                cache_key,
                lambda: ort.InferenceSession(encoder_model_path, providers=[provider]),
            )
        except Exception as e:
            logger.error(f"Failed to create ONNX session for {encoder_model_path}: {e}")
            return None

    @staticmethod
    def _preprocess_text(text: str, prepend_text: Optional[str] = None) -> str:
        """Prepend text if provided."""
        return f"{prepend_text}{text}" if prepend_text else text

    @staticmethod
    def _log_onnx_outputs(outputs: Any, session: Optional[Any]) -> None:
        """Log ONNX outputs in debug mode."""
        if not APP_SETTINGS.app.debug or not outputs:
            return

        output_names = (
            [o.name for o in session.get_outputs()]
            if session
            else [f"output_{i}" for i in range(len(outputs))]
        )

        for name, arr in zip(output_names, outputs):
            logger.debug(f"ONNX output: {name}, shape: {arr.shape}, dtype: {arr.dtype}")

    @staticmethod
    def _is_logits_output(outputs: Any, session: Optional[Any] = None) -> bool:
        """Fast logits detection."""
        if not outputs:
            return False

        # Check output names first
        if session:
            output_names = [o.name.lower() for o in session.get_outputs()]
            if any(
                keyword in name
                for name in output_names
                for keyword in ["logit", "score", "prob"]
            ):
                return True

        # Shape heuristic - likely vocab size
        arr = outputs[0]
        return arr.ndim >= 2 and arr.shape[-1] <= 50000

    @staticmethod
    def clear_thread_tokenizers() -> None:
        """Clear thread-local tokenizer cache."""
        if hasattr(_tokenizer_local, "tokenizers"):
            _tokenizer_local.tokenizers.clear()
