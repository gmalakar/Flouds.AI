# =============================================================================
# File: base_nlp_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import threading
from typing import Any, Optional

import onnxruntime as ort
from numpy import ndarray
from transformers import AutoTokenizer, PreTrainedTokenizer

from app.app_init import APP_SETTINGS
from app.config.config_loader import ConfigLoader
from app.exceptions import InvalidConfigError, ModelLoadError, TokenizerError
from app.logger import get_logger
from app.modules.concurrent_dict import ConcurrentDict
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.simple_cache import SimpleCache

logger = get_logger("base_nlp_service")

# Thread-local storage for tokenizers
_tokenizer_local = threading.local()

# Model configuration cache
_model_config_cache = SimpleCache(max_size=50)


class BaseNLPService:
    """
    Base class for NLP services providing thread-safe tokenizer/session management,
    configuration loading, and utility methods for ONNX-based inference.
    """

    _root_path: str = APP_SETTINGS.onnx.onnx_path
    _CACHE_LIMIT_ENCODER = int(os.getenv("FLOUDS_ENCODER_CACHE_MAX", "3"))

    _encoder_sessions: ConcurrentDict = ConcurrentDict(
        "_encoder_sessions", max_size=_CACHE_LIMIT_ENCODER
    )
    _model_cache: SimpleCache = SimpleCache(max_size=5)

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        """Numerically stable softmax."""
        import numpy as np

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
        Load ONNX model configuration with caching.
        """
        # Check cache first
        cached_config = _model_config_cache.get(model_to_use)
        if cached_config is not None:
            return cached_config

        try:
            config = ConfigLoader.get_onnx_config(model_to_use)
            if not BaseNLPService._validate_model_config(config):
                logger.error(
                    "Invalid config for model '%s': missing required fields",
                    sanitize_for_log(model_to_use),
                )
                return None

            # Cache the valid config
            _model_config_cache.put(model_to_use, config)
            return config

        except (KeyError, AttributeError) as e:
            logger.error(
                "Model config '%s' not found in onnx_config.json: %s",
                sanitize_for_log(model_to_use),
                sanitize_for_log(str(e)),
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to load config for model '%s': %s",
                sanitize_for_log(model_to_use),
                sanitize_for_log(str(e)),
            )
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
                            tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_path, local_files_only=True, legacy=True
                            )
                        else:
                            tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_path, local_files_only=True
                            )
                    except (OSError, ValueError) as ex:
                        logger.warning(
                            "Fallback to legacy tokenizer for %s: %s",
                            sanitize_for_log(tokenizer_path),
                            sanitize_for_log(str(ex)),
                        )
                        tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_path, local_files_only=True, legacy=True
                        )
                    except Exception as ex:
                        if "PyPreTokenizerTypeWrapper" in str(ex):
                            logger.warning(
                                "PyPreTokenizerTypeWrapper error, trying from HuggingFace for %s",
                                sanitize_for_log(tokenizer_path),
                            )
                            # Extract model name from path for HuggingFace download
                            model_name = os.path.basename(tokenizer_path)
                            if model_name in ["sentence-t5-base", "all-MiniLM-L6-v2"]:
                                tokenizer = AutoTokenizer.from_pretrained(
                                    f"sentence-transformers/{model_name}", legacy=True
                                )
                            else:
                                raise ex
                        else:
                            raise ex
                    except Exception as ex:
                        logger.error(
                            "Failed to load tokenizer for %s: %s",
                            sanitize_for_log(tokenizer_path),
                            sanitize_for_log(str(ex)),
                        )
                        raise TokenizerError(f"Cannot load tokenizer: {ex}")
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

                _tokenizer_local.tokenizers[cache_key] = tokenizer

            return _tokenizer_local.tokenizers[cache_key]

        except (OSError, FileNotFoundError) as e:
            logger.error(
                "Tokenizer files not accessible at %s: %s",
                sanitize_for_log(tokenizer_path),
                sanitize_for_log(str(e)),
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to load tokenizer from %s: %s",
                sanitize_for_log(tokenizer_path),
                sanitize_for_log(str(e)),
            )
            raise TokenizerError(f"Tokenizer loading failed: {e}")

    @staticmethod
    def _get_encoder_session(encoder_model_path: str) -> Optional[ort.InferenceSession]:
        """
        Get or create ONNX session with error handling and fallback to regular model.
        Adds provider to cache key for multi-provider support.
        """
        try:
            provider = APP_SETTINGS.server.session_provider or "CPUExecutionProvider"

            # Validate provider availability
            available_providers = ort.get_available_providers()
            if provider not in available_providers:
                logger.warning(
                    "Provider %s not available, using CPUExecutionProvider",
                    sanitize_for_log(provider),
                )
                provider = "CPUExecutionProvider"

            cache_key = f"{encoder_model_path}#{provider}"

            # First check if session is already cached
            cached_session = BaseNLPService._encoder_sessions.get(cache_key)
            if cached_session is not None:
                return cached_session

            # Session not in cache - check memory and clear if needed
            from app.utils.cache_manager import CacheManager

            CacheManager.check_and_clear_cache_if_needed()

            def create_session():
                return ort.InferenceSession(encoder_model_path, providers=[provider])

            return BaseNLPService._encoder_sessions.get_or_add(
                cache_key, create_session
            )
        except (OSError, FileNotFoundError) as e:
            logger.error(
                "ONNX model file not accessible at %s: %s",
                sanitize_for_log(encoder_model_path),
                sanitize_for_log(str(e)),
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to create ONNX session for %s: %s",
                sanitize_for_log(encoder_model_path),
                sanitize_for_log(str(e)),
            )
            raise ModelLoadError(f"ONNX session creation failed: {e}")

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
    def clear_model_config_cache() -> None:
        """Clear model configuration cache."""
        _model_config_cache.clear()
        logger.info("Model configuration cache cleared")

    @staticmethod
    def get_cache_stats() -> dict:
        """Get cache statistics for monitoring."""
        return {
            "encoder_sessions": BaseNLPService._encoder_sessions.size(),
            "model_configs": _model_config_cache.size(),
            "model_cache": BaseNLPService._model_cache.size(),
        }

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
            logger.debug(
                "ONNX output %s: shape=%s",
                sanitize_for_log(name),
                sanitize_for_log(str(arr.shape)),
            )

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

    @staticmethod
    def warm_up_cache(model_names: list[str]) -> None:
        """Pre-load model configurations into cache."""
        logger.info(f"Warming up cache for {len(model_names)} models")
        for model_name in model_names:
            try:
                BaseNLPService._get_model_config(model_name)
            except (InvalidConfigError, TokenizerError, ModelLoadError) as e:
                logger.warning(f"Failed to warm up cache for {model_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error warming up cache for {model_name}: {e}")
                raise InvalidConfigError(f"Cache warm-up failed: {e}")
        logger.info("Cache warm-up completed")
