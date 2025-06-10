# =============================================================================
# File: base_nlp_service.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import threading
from typing import Any, Dict, Optional

import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizer

from app.config.config_loader import ConfigLoader
from app.logger import get_logger
from app.modules.concurrent_dict import ConcurrentDict
from app.setup import APP_SETTINGS

logger = get_logger("base_nlp_service")

# Add thread-local storage for tokenizers
_tokenizer_local = threading.local()


class BaseNLPService:
    _root_path: str = os.path.abspath(
        APP_SETTINGS.onnx.rootpath
        if APP_SETTINGS.onnx.rootpath
        else os.path.join(os.path.dirname(__file__), "..", "onnx")
    )
    _encoder_sessions: ConcurrentDict = ConcurrentDict("_encoder_sessions")

    logger.debug(f"Base NLP service root path: {_root_path}")

    @staticmethod
    def _get_model_config(model_to_use: str) -> Any:
        """
        Load and return the ONNX model configuration for the given model name.
        """
        logger.debug(f"Loading model config for: {model_to_use}")
        return ConfigLoader.get_onnx_config(model_to_use)

    @staticmethod
    def _get_tokenizer_threadsafe(tokenizer_path: str) -> PreTrainedTokenizer:
        """
        Returns a thread-local tokenizer instance for the given path.
        Ensures no concurrency issues with HuggingFace tokenizers.
        """
        logger.debug(f"Getting tokenizer for path: {tokenizer_path}")
        # Thread-safe: one tokenizer per thread per path
        if not hasattr(_tokenizer_local, "tokenizers"):
            _tokenizer_local.tokenizers = {}
        if tokenizer_path not in _tokenizer_local.tokenizers:
            _tokenizer_local.tokenizers[tokenizer_path] = AutoTokenizer.from_pretrained(
                tokenizer_path
            )
        return _tokenizer_local.tokenizers[tokenizer_path]

    # Optionally, for backward compatibility:
    _get_tokenizer = _get_tokenizer_threadsafe

    @staticmethod
    def _get_encoder_session(encoder_model_path: str) -> ort.InferenceSession:
        """
        Get or create an ONNX InferenceSession for the encoder model at the given path.
        """
        provider = APP_SETTINGS.server.model_session_provider or "CPUExecutionProvider"
        logger.debug(
            f"Getting ONNX encoder session for path: {encoder_model_path} with provider: {provider}"
        )
        providers = [provider]
        return BaseNLPService._encoder_sessions.get_or_add(
            (encoder_model_path, provider),
            lambda: ort.InferenceSession(encoder_model_path, providers=providers),
        )

    @staticmethod
    def _preprocess_text(text: str, prepend_text: Optional[str] = None) -> str:
        """
        Prepend text if provided, otherwise return the original text.
        """
        return f"{prepend_text}{text}" if prepend_text else text

    @staticmethod
    def _log_onnx_outputs(outputs: Any, session: Optional[Any]) -> None:
        """
        Log ONNX model outputs for debugging if app debug mode is enabled.
        """
        if APP_SETTINGS.app.debug:
            if session is not None:
                output_names = [o.name for o in session.get_outputs()]
            else:
                output_names = [f"output_{i}" for i in range(len(outputs))]
            for name, arr in zip(output_names, outputs):
                logger.debug(
                    f"ONNX output: {name}, shape: {arr.shape}, dtype: {arr.dtype}"
                )

    @staticmethod
    def _is_logits_output(outputs: Any, session: Optional[Any] = None) -> bool:
        """
        Heuristically determine if the ONNX output is logits.
        """
        if session is not None:
            output_names = [o.name.lower() for o in session.get_outputs()]
            for name in output_names:
                if "logit" in name or "score" in name or "prob" in name:
                    return True
        arr = outputs[0]
        shape = arr.shape
        if len(shape) == 2:
            if shape[1] <= 10:
                return True
        elif len(shape) == 3:
            if shape[2] <= 10:
                return True
        return False

    @staticmethod
    def clear_thread_tokenizers() -> None:
        """
        Clear the thread-local tokenizer cache for the current thread.
        Useful for memory management or when reloading models.
        """
        if hasattr(_tokenizer_local, "tokenizers"):
            _tokenizer_local.tokenizers.clear()
