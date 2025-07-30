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
from app.utils.model_cache import LRUModelCache

logger = get_logger("base_nlp_service")

# Thread-local storage for tokenizers
_tokenizer_local = threading.local()


class BaseNLPService:
    """
    Base class for NLP services providing thread-safe tokenizer/session management,
    configuration loading, and utility methods for ONNX-based inference.
    """

    _root_path: str = APP_SETTINGS.onnx.onnx_path
    _encoder_sessions: ConcurrentDict = ConcurrentDict("_encoder_sessions")
    _model_cache: LRUModelCache = LRUModelCache(max_size=5, ttl_seconds=3600)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = np.asarray(x)
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def _validate_model_config(config: Any) -> bool:
        """Validate required config fields are present."""
        if config is None:
            return False
        required_fields = ["inputnames", "outputnames"]
        return all(hasattr(config, field) for field in required_fields)

    @staticmethod
    def _get_model_config(model_to_use: str) -> Any:
        """
        Load ONNX model configuration.
        Logs config path and errors if loading fails.
        """
        try:
            config = ConfigLoader.get_onnx_config(model_to_use)
            if not BaseNLPService._validate_model_config(config):
                logger.error(
                    f"Invalid config for model '{model_to_use}': missing required fields"
                )
                return None
            return config
        except Exception as e:
            logger.error(f"Failed to load config for model '{model_to_use}': {e}")
            return None

    @staticmethod
    def _get_tokenizer_threadsafe(
        tokenizer_path: str, use_legacy: bool = False
    ) -> Optional[PreTrainedTokenizer]:
        """
        Thread-safe tokenizer loading with error handling.
        Logs which fallback path was used.
        """
        try:
            if not hasattr(_tokenizer_local, "tokenizers"):
                _tokenizer_local.tokenizers = {}

            cache_key = f"{tokenizer_path}#{use_legacy}"
            if cache_key not in _tokenizer_local.tokenizers:
                if os.path.exists(tokenizer_path):
                    try:
                        if use_legacy:
                            logger.debug(
                                f"Loading legacy tokenizer from {tokenizer_path}"
                            )
                            tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_path, local_files_only=True, legacy=True
                            )
                        else:
                            logger.debug(f"Loading tokenizer from {tokenizer_path}")
                            tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_path, local_files_only=True
                            )
                    except Exception as ex:
                        logger.warning(
                            f"Fallback to legacy tokenizer for {tokenizer_path}: {ex}"
                        )
                        tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_path, local_files_only=True, legacy=True
                        )
                else:
                    logger.debug(
                        f"Loading tokenizer from HuggingFace Hub: {tokenizer_path}"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

                _tokenizer_local.tokenizers[cache_key] = tokenizer

            return _tokenizer_local.tokenizers[cache_key]

        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            return None

    @staticmethod
    def _get_encoder_session(encoder_model_path: str) -> Optional[ort.InferenceSession]:
        """
        Get or create ONNX session with error handling.
        Adds provider to cache key for multi-provider support.
        """
        try:
            provider = APP_SETTINGS.server.session_provider or "CPUExecutionProvider"

            # Validate provider availability
            available_providers = ort.get_available_providers()
            if provider not in available_providers:
                logger.warning(
                    f"Provider {provider} not available, using CPUExecutionProvider"
                )
                provider = "CPUExecutionProvider"

            cache_key = f"{encoder_model_path}#{provider}"

            return BaseNLPService._encoder_sessions.get_or_add(
                cache_key,
                lambda: ort.InferenceSession(encoder_model_path, providers=[provider]),
            )
        except Exception as e:
            logger.error(f"Failed to create ONNX session for {encoder_model_path}: {e}")
            return None

    @staticmethod
    def clear_encoder_sessions() -> None:
        """Clear cached ONNX encoder sessions (useful for testing/reloading)."""
        BaseNLPService._encoder_sessions.clear()

    @staticmethod
    def clear_thread_tokenizers() -> None:
        """Clear thread-local tokenizer cache."""
        if hasattr(_tokenizer_local, "tokenizers"):
            _tokenizer_local.tokenizers.clear()

    @staticmethod
    def _prepend_text(text: str, prepend_text: Optional[str] = None) -> str:
        """
        Prepend text if provided, ensuring both are strings.
        """
        if prepend_text is not None:
            return f"{str(prepend_text)}{str(text)}"
        return str(text)

    @staticmethod
    def _log_onnx_outputs(outputs: Any, session: Optional[Any]) -> None:
        """
        Log ONNX outputs in debug mode.
        Logs output shapes, dtypes, and a sample of values for deeper debugging.
        """
        if not APP_SETTINGS.app.debug or not outputs:
            return

        output_names = (
            [o.name for o in session.get_outputs()]
            if session
            else [f"output_{i}" for i in range(len(outputs))]
        )

        for name, arr in zip(output_names, outputs):
            logger.debug(f"ONNX output: {name}, shape: {arr.shape}, dtype: {arr.dtype}")
            # Log a sample of output values for debugging
            if arr.size > 0:
                logger.debug(f"Sample values for {name}: {arr.flatten()[:5]}")

    @staticmethod
    def _is_logits_output(
        outputs: Any, session: Optional[Any] = None, vocab_threshold: int = 50000
    ) -> bool:
        """
        Fast logits detection.
        Checks output names and shape heuristics.
        Vocab size threshold is configurable.
        """
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
        return arr.ndim >= 2 and arr.shape[-1] <= vocab_threshold
