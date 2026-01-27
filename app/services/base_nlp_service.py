# =============================================================================
# File: base_nlp_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np
import onnxruntime as ort
from numpy import ndarray
from transformers import AutoTokenizer, PreTrainedTokenizer

from app.app_init import APP_SETTINGS
from app.config.config_loader import ConfigLoader
from app.exceptions import (
    InvalidConfigError,
    InvalidInputError,
    MissingConfigError,
    ModelLoadError,
    ModelNotFoundError,
    TokenizerError,
)
from app.logger import get_logger
from app.models.model_metadata import ModelMetadata
from app.modules.concurrent_dict import ConcurrentDict
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.path_validator import validate_safe_path
from app.utils.simple_cache import SimpleCache

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
except Exception:
    ORTModelForSeq2SeqLM = None

# Cache primitives moved to `cache_registry` to decouple CacheManager
from app.services.cache_registry import MODEL_CONFIG_CACHE
from app.services.cache_registry import clear_encoder_sessions as cache_clear_encoder_sessions
from app.services.cache_registry import clear_model_config_cache as cache_clear_model_config_cache
from app.services.cache_registry import clear_thread_tokenizers as cache_clear_thread_tokenizers
from app.services.cache_registry import get_cache_stats as cache_get_cache_stats
from app.services.cache_registry import (
    get_decoder_sessions,
    get_encoder_sessions,
    get_models_cache,
    tokenizer_local,
)

logger = get_logger("base_nlp_service")

# HuggingFace model mappings for fallback
HUGGINGFACE_MODEL_MAPPING = {
    "sentence-t5-base": "sentence-transformers/sentence-t5-base",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


class BaseNLPService:
    """
    Base class for NLP services providing thread-safe tokenizer/session management,
    configuration loading, and utility methods for ONNX-based inference.
    """

    # Ensure root path is a string to satisfy static type expectations
    _root_path: str = str(getattr(APP_SETTINGS.onnx, "onnx_path", ""))
    _CACHE_LIMIT_ENCODER = int(os.getenv("FLOUDS_ENCODER_CACHE_MAX", "3"))
    _encoder_sessions: ConcurrentDict = ConcurrentDict(
        "_encoder_sessions", max_size=_CACHE_LIMIT_ENCODER
    )
    # Small metadata cache for quick model metadata lookups (resolved path,
    # last-loaded timestamp, lightweight flags). Heavy objects (sessions,
    # models) remain in the ConcurrentDict caches above.
    _model_metadata_cache: SimpleCache = SimpleCache(max_size=50)
    # Small set of validated resolved model folder paths to avoid repeated
    # expensive `validate_safe_path` calls. Use a lock for simple thread-safety.
    _validated_model_paths: Set[str] = set()
    _validated_paths_lock = threading.Lock()

    # Shared caches for other services (decoder sessions, optimum models,
    # and special tokens). Centralized here so `CacheManager` can operate
    # across all services consistently.
    pass

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
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
        # inputnames and outputnames can now be auto-detected, so not strictly required
        # Only fail if config is completely empty or invalid
        return True

    @staticmethod
    def _get_model_config(model_to_use: str) -> Any:
        """
        Load ONNX model configuration with caching.
        """
        # Check cache first
        cached_config = MODEL_CONFIG_CACHE.get(model_to_use)
        if cached_config is not None:
            return cached_config

        try:
            # Require exact key match when loading model config. This avoids
            # silently accepting variant names (e.g. 't5_small') which could
            # mask typos or incorrect model identifiers.
            config = ConfigLoader.get_onnx_config(model_to_use)
            if not BaseNLPService._validate_model_config(config):
                logger.error(
                    "Invalid config for model '%s': missing required fields",
                    sanitize_for_log(model_to_use),
                )
                return None

            # Cache the valid config
            MODEL_CONFIG_CACHE.put(model_to_use, config)
            return config
        except (KeyError, AttributeError, MissingConfigError) as e:
            logger.error(
                "Failed to load config for model '%s': %s",
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
    def _get_model_path(model_to_use: str) -> Optional[str]:
        """Resolve the full model folder path for a given model name.

        Preference order:
        - `OnnxConfig.model_folder_name` when present. If it points to a parent
          folder (e.g. 'f2') the actual folder used will be 'f2/<model_to_use>'.
          If it already includes the model name (e.g. 'f2/all-mpnet-base-v2')
          it will be used as-is.
        - If `model_folder_name` is not set, prefer `summarization_task` when it
          is present and equals 's2s' or 'llm' (so seq2seq models go under
          's2s/<model>' or 'llm/<model>'). Otherwise fall back to
          `embedder_task` (default 'fe').

        Returns the absolute validated path or None if config missing/invalid.
        """
        try:
            config = BaseNLPService._get_model_config(model_to_use)
            if not config:
                return None

            # Require explicit model_folder_name - no implicit fallback
            mf = getattr(config, "model_folder_name", None)
            if not mf:
                # Do not attempt to infer folder from other keys; caller should
                # provide `model_folder_name` in the config. Return None so
                # callers can surface an appropriate error.
                logger.error("Model '%s' missing 'model_folder_name' in config", model_to_use)
                return None

            mf_norm = os.path.normpath(str(mf))
            # Determine whether the configured folder already includes the
            # model name. Support common variants (hyphen vs underscore)
            # so that keys like 't5_small' correctly map to folders named
            # 't5-small' without appending the model name again.
            basename = os.path.basename(mf_norm)
            model_variants = {
                model_to_use,
                str(model_to_use).replace("_", "-"),
                str(model_to_use).replace("-", "_"),
            }
            if basename in model_variants:
                rel_folder = mf_norm
            else:
                rel_folder = os.path.join(mf_norm, model_to_use)

            full_path = validate_safe_path(
                os.path.join(BaseNLPService._root_path, "models", rel_folder),
                BaseNLPService._root_path,
            )
            logger.debug("Resolved model path for %s -> %s", model_to_use, full_path)
            # Populate lightweight metadata for quick lookups (resolved path, folder, tasks)
            try:
                metadata = {
                    "resolved_path": full_path,
                    "validated_resolved_path": full_path,
                    "model_folder_name": mf_norm,
                    "tasks": getattr(config, "tasks", None),
                    "timestamp": int(time.time()),
                }
                BaseNLPService._set_model_metadata(model_to_use, metadata)
                # Record validated path in class-level cache so subsequent
                # file-existence checks can skip path validation.
                try:
                    with BaseNLPService._validated_paths_lock:
                        BaseNLPService._validated_model_paths.add(full_path)
                except Exception:
                    # Non-fatal - don't break path resolution on lock failures
                    pass
            except Exception:
                # Metadata population shouldn't block normal flow
                logger.debug(f"Failed to set metadata for {model_to_use}")
            return full_path
        except Exception as e:
            logger.error(
                "Error resolving model path for %s: %s",
                sanitize_for_log(model_to_use),
                sanitize_for_log(str(e)),
            )
            return None

    @staticmethod
    def _file_exists_in_model(folder: str, *candidates: str) -> bool:
        """Check whether any of the candidate filenames exist under `folder`.

        Uses `validate_safe_path` to ensure paths are safe and returns True
        if any candidate resolves to an existing file. Skips empty/None
        candidates.
        """
        # Fast path: if this exact folder was validated previously, reuse it
        # without re-calling `validate_safe_path` which can be expensive.
        validated_folder = None
        try:
            with BaseNLPService._validated_paths_lock:
                if folder in BaseNLPService._validated_model_paths:
                    validated_folder = folder
        except Exception:
            validated_folder = None

        # If not cached, validate the folder once and record the validated
        # result for future checks. On validation failure fall back to the
        # previous per-candidate validation behavior.
        if validated_folder is None:
            try:
                validated_folder = validate_safe_path(folder, BaseNLPService._root_path)
                try:
                    with BaseNLPService._validated_paths_lock:
                        BaseNLPService._validated_model_paths.add(validated_folder)
                except Exception:
                    pass
            except Exception:
                # If folder validation fails, fall back to per-candidate validation
                for candidate in candidates:
                    if not candidate:
                        continue
                    try:
                        candidate_path = validate_safe_path(
                            os.path.join(folder, candidate), BaseNLPService._root_path
                        )
                    except Exception:
                        continue
                    if os.path.exists(candidate_path):
                        return True
                return False

        for candidate in candidates:
            if not candidate:
                continue
            try:
                candidate_path = os.path.join(validated_folder, candidate)
            except Exception:
                continue
            if os.path.exists(candidate_path):
                return True
        return False

    @staticmethod
    def file_exists_in_model(folder: str, *candidates: str) -> bool:
        """Public compatibility wrapper for file existence checks inside a model folder."""
        return BaseNLPService._file_exists_in_model(folder, *candidates)

    @staticmethod
    def _get_model_filename(
        model_config: Any,
        model_type: str = "encoder",
        fallback_to_regular: bool = False,
    ) -> str:
        """Return the preferred ONNX filename for a given model config and model type.

        model_type: 'encoder' or 'decoder' (or 'embedding' treated as encoder).
        The project no longer supports a `use_optimized` flag; prefer the
        canonical filename fields and fall back to optimized variants when
        a primary name is not present.
        """

        # Normalize type
        t = model_type.lower()
        if t in ("embedding", "encoder"):
            # No `use_optimized` support: always prefer canonical filename fields.
            return getattr(model_config, "encoder_onnx_model", None) or "model.onnx"
        else:
            return getattr(model_config, "decoder_onnx_model", None) or "decoder_model.onnx"

    @staticmethod
    def _validate_model_availability(
        model_to_use: str,
        task: Optional[str] = None,
        perform_filesystem_check: bool = True,
        cfg: Optional[Any] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        """Validate that a model exists in config and that required model files exist.

        - Ensures config is present for `model_to_use`.
        - Resolves model folder via `_get_model_path`.
        - Checks presence of encoder (and decoder when needed) ONNX files.

        `task` can be 'embedding' or 'prompt' (affects which files are required).
        Returns True when checks pass, False otherwise.
        """
        try:
            # Allow caller to provide a pre-fetched config (useful for tests
            # that patch class-level getters). If not provided, fetch from
            # centralized loader.
            if cfg is None:
                cfg = BaseNLPService._get_model_config(model_to_use)
            if not cfg:
                logger.error("Model config '%s' not found", sanitize_for_log(model_to_use))
                return False

            # Allow caller to provide an already-resolved model_path (useful
            # when the caller has a patched `_get_model_path` or wants to
            # avoid an extra lookup). Otherwise resolve via centralized
            # resolver.
            if model_path is None:
                model_path = BaseNLPService._get_model_path(model_to_use)
            if not model_path:
                logger.error(
                    "Model path for '%s' could not be resolved",
                    sanitize_for_log(model_to_use),
                )
                return False

            # Decide which artifacts we need to verify based on the requested task
            # Encoder is required for embedding and prompt flows; decoder only
            # required when seq2seq generation is expected.
            need_encoder = True
            need_decoder = False
            if task == "prompt":
                if getattr(cfg, "use_seq2seqlm", False):
                    need_decoder = True
                elif not getattr(cfg, "encoder_only", False):
                    need_decoder = True

            # Short-circuit using cached flags stored in the model metadata
            # cache to avoid filesystem IO when possible. Flags are expected
            # to be one of 'not_checked' | 'true' | 'false'.
            md = BaseNLPService._get_model_metadata(model_to_use) or {}
            enc_flag = md.get("encoder_model_exists", "not_checked")
            dec_flag = md.get("decoder_model_exists", "not_checked")

            # If caller requests to skip filesystem checks, only use cached
            # flags and basic config/path validation. This is useful for
            # test environments or callers that will perform stricter
            # validation later when actually loading sessions.
            if not perform_filesystem_check:
                if need_encoder and enc_flag == "false":
                    logger.error(
                        "Encoder ONNX model (cached=false) missing for '%s' in %s",
                        sanitize_for_log(model_to_use),
                        sanitize_for_log(model_path),
                    )
                    return False
                if need_decoder and dec_flag == "false":
                    logger.error(
                        "Decoder ONNX model (cached=false) missing for '%s' in %s",
                        sanitize_for_log(model_to_use),
                        sanitize_for_log(model_path),
                    )
                    return False
                return True

            # If encoder already known present and decoder either not required
            # or known present, we can return early.
            if need_encoder and enc_flag == "true" and (not need_decoder or dec_flag == "true"):
                return True

            # If any required artifact is cached as missing, fail fast.
            if need_encoder and enc_flag == "false":
                logger.error(
                    "Encoder ONNX model (cached=false) missing for '%s' in %s",
                    sanitize_for_log(model_to_use),
                    sanitize_for_log(model_path),
                )
                return False
            if need_decoder and dec_flag == "false":
                logger.error(
                    "Decoder ONNX model (cached=false) missing for '%s' in %s",
                    sanitize_for_log(model_to_use),
                    sanitize_for_log(model_path),
                )
                return False

            # For any artifact still `not_checked`, perform checks now. Build
            # candidate lists lazily only for those artifacts.
            file_exists_in_model = BaseNLPService._file_exists_in_model

            # Track final results; default to True and falsify on failure
            final_ok = True

            # Encoder check (only if not already known true)
            if need_encoder and enc_flag != "true":
                enc_candidates = []
                try:
                    enc_cand = BaseNLPService._get_model_filename(cfg, "encoder")
                except Exception:
                    enc_cand = None
                if enc_cand:
                    enc_candidates.append(enc_cand)
                enc_candidates.extend(
                    [
                        getattr(cfg, "encoder_onnx_model", "model.onnx"),
                        "encoder_model.onnx",
                    ]
                )

                exists = file_exists_in_model(model_path, *enc_candidates)
                # Store result in local metadata map; will persist below
                md["encoder_model_exists"] = "true" if exists else "false"
                if not exists:
                    logger.error(
                        "Encoder ONNX model missing for '%s' in %s",
                        sanitize_for_log(model_to_use),
                        sanitize_for_log(model_path),
                    )
                    final_ok = False

            # Auto-detect encoder-only models: when a decoder is expected but
            # only a canonical single-file encoder (`model.onnx`) exists, treat
            # the model as encoder-only so callers won't require a separate
            # `decoder_model.onnx` file. This helps support LLM exports that
            # produce a single `model.onnx` artifact for encoder-only runs.
            try:
                if need_decoder and dec_flag != "true":
                    has_canonical_model = BaseNLPService._file_exists_in_model(
                        model_path,
                        getattr(cfg, "encoder_onnx_model", "model.onnx"),
                        "model.onnx",
                    )
                    has_decoder_files = BaseNLPService._file_exists_in_model(
                        model_path,
                        getattr(cfg, "decoder_onnx_model", "decoder_model.onnx"),
                    )
                    # Only infer encoder-only when the model is not explicitly
                    # configured as a seq2seq LM. If `use_seq2seqlm` is True we
                    # must require decoder artifacts.
                    if (
                        has_canonical_model
                        and not has_decoder_files
                        and not getattr(cfg, "use_seq2seqlm", False)
                    ):
                        logger.info(
                            "Inferring encoder-only model for '%s' (found model.onnx, no decoder artifacts)",
                            sanitize_for_log(model_to_use),
                        )
                        md["decoder_model_exists"] = "false"
                        md["encoder_model_exists"] = "true"
                        md["encoder_only_inferred"] = True
                        # Disable decoder requirement so subsequent checks pass
                        need_decoder = False
            except Exception:
                # Non-fatal: inference failure shouldn't block availability checks
                logger.debug("Encoder-only inference check failed", exc_info=True)

            # Decoder check (only if required and not already known true)
            if need_decoder and dec_flag != "true":
                dec_candidates = []
                try:
                    dec_cand = BaseNLPService._get_model_filename(cfg, "decoder")
                except Exception:
                    dec_cand = None
                if dec_cand:
                    dec_candidates.append(dec_cand)
                dec_candidates.extend([getattr(cfg, "decoder_onnx_model", "decoder_model.onnx")])

                dec_exists = BaseNLPService._file_exists_in_model(model_path, *dec_candidates)
                # Store result in local metadata map; will persist below
                md["decoder_model_exists"] = "true" if dec_exists else "false"
                if not dec_exists:
                    logger.error(
                        "Decoder ONNX model missing for '%s' in %s",
                        sanitize_for_log(model_to_use),
                        sanitize_for_log(model_path),
                    )
                    final_ok = False

            # Update metadata cache with final existence flags and last-checked timestamp
            try:
                # Persist metadata flags and last-checked timestamp
                md.update({"last_checked": int(time.time())})
                BaseNLPService._set_model_metadata(model_to_use, md)
            except Exception:
                logger.debug(f"Failed to update metadata for {model_to_use}")

            return final_ok
        except Exception as e:
            logger.error(
                "Error validating availability for %s: %s",
                sanitize_for_log(model_to_use),
                sanitize_for_log(str(e)),
            )
            return False

    @staticmethod
    def _can_perform_task(model_name: str, task: str) -> bool:
        """Return True if the named model declares support for the given task.

        Checks the model's `tasks` list (normalized to lower-case). If `tasks`
        is not present, returns False. This method now validates that the
        model configuration exists and will raise `ModelNotFoundError` when
        the requested `model_name` is unknown.

        Examples of `task` values: 'embedding', 'prompt', 'summarization',
        'language_model', 'llm'
        """
        try:
            if not model_name or not task:
                raise InvalidInputError("Both 'model_name' and 'task' are required parameters")

            # Validate that the model configuration exists. Raise a
            # ModelNotFoundError so callers can surface a clear error when
            # an invalid model name is provided.
            cfg = BaseNLPService._get_model_config(model_name)
            if not cfg:
                raise ModelNotFoundError(f"Model '{model_name}' not found")

            # Prefer explicit tasks list when available
            if hasattr(cfg, "tasks") and cfg.tasks:
                try:
                    tasks = [str(t).lower() for t in cfg.tasks]
                except Exception:
                    tasks = []
                return str(task).lower() in tasks

            # Require explicit `tasks` in model config; do not fall back to
            # legacy fields here.
            return False
        except ModelNotFoundError:
            # Propagate explicit not-found errors so callers can handle them.
            raise
        except Exception:
            return False

    @staticmethod
    def _get_tokenizer_threadsafe(
        tokenizer_path: str, use_legacy: bool = False
    ) -> Optional[PreTrainedTokenizer]:
        """
        Thread-safe tokenizer loading with error handling.
        Logs which fallback path was used.
        """
        try:
            if not hasattr(tokenizer_local, "tokenizers"):
                tokenizer_local.tokenizers = {}

            # Normalize tokenizer path/id for stable cache keys. If the
            # tokenizer_path points to an existing filesystem path, use the
            # absolute normalized path; otherwise keep the provided identifier
            # as a string. Fall back to `repr()` for unexpected types. Include
            # the legacy flag explicitly so the cache distinguishes modes.
            try:
                if isinstance(tokenizer_path, (str, bytes, os.PathLike)):
                    tpath = str(tokenizer_path)
                    if os.path.exists(tpath):
                        tpath = os.path.abspath(os.path.normpath(tpath))
                else:
                    # Non-path-like objects: use a reproducible representation
                    tpath = repr(tokenizer_path)
            except Exception:
                tpath = str(tokenizer_path)

            cache_key = f"{tpath}#legacy={1 if use_legacy else 0}"
            if cache_key not in tokenizer_local.tokenizers:
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
                            hf_model = HUGGINGFACE_MODEL_MAPPING.get(model_name)
                            if hf_model:
                                tokenizer = AutoTokenizer.from_pretrained(hf_model, legacy=True)
                            else:
                                raise TokenizerError(
                                    f"No HuggingFace mapping found for {model_name}"
                                )
                        else:
                            logger.error(
                                "Failed to load tokenizer for %s: %s",
                                sanitize_for_log(tokenizer_path),
                                sanitize_for_log(str(ex)),
                            )
                            raise TokenizerError(f"Cannot load tokenizer: {ex}")
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

                tokenizer_local.tokenizers[cache_key] = tokenizer

            return tokenizer_local.tokenizers[cache_key]

        except OSError as e:
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
    def _get_tokenizer(
        tokenizer_path: str, use_legacy: bool = False
    ) -> Optional[PreTrainedTokenizer]:
        """Public wrapper for thread-safe tokenizer loading.

        Centralizes tokenizer loading entrypoint so callers and tests can
        mock a single symbol (`BaseNLPService.get_tokenizer`) instead of
        referencing the internal method.
        """
        return BaseNLPService._get_tokenizer_threadsafe(tokenizer_path, use_legacy)

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

            # Normalize the model path to produce stable cache keys
            try:
                normalized_path = os.path.abspath(os.path.normpath(encoder_model_path))
            except Exception:
                normalized_path = str(encoder_model_path)
            cache_key = f"{normalized_path}#{provider}"

            # Ensure registry caches initialized and cast for static checkers
            encoder_sessions = get_encoder_sessions()

            # First check if session is already cached
            cached_session = encoder_sessions.get(cache_key)
            if cached_session is not None:
                return cached_session

            # Session not in cache - check memory and clear if needed
            from app.utils.cache_manager import CacheManager

            CacheManager.check_and_clear_cache_if_needed()

            def create_session():
                return ort.InferenceSession(encoder_model_path, providers=[provider])

            return encoder_sessions.get_or_add(cache_key, create_session)
        except OSError as e:
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
    def _clear_encoder_sessions() -> None:
        """Clear cached ONNX encoder sessions (useful for testing/reloading)."""
        cache_clear_encoder_sessions()

    @staticmethod
    def clear_encoder_sessions() -> None:
        """Public wrapper to clear ONNX encoder sessions (compat)."""
        return BaseNLPService._clear_encoder_sessions()

    @staticmethod
    def _get_decoder_session(decoder_model_path: str) -> Optional[ort.InferenceSession]:
        """Get cached ONNX decoder session with error handling and provider support.

        Public wrapper so all callers use the centralized `_decoder_sessions` cache
        and consistent memory-check logic.
        """
        try:
            provider = APP_SETTINGS.server.session_provider or "CPUExecutionProvider"
            available_providers = ort.get_available_providers()
            if provider not in available_providers:
                logger.warning(
                    "Provider %s not available for decoder, using CPUExecutionProvider",
                    sanitize_for_log(provider),
                )
                provider = "CPUExecutionProvider"

            # Normalize the decoder path for stable cache keys
            try:
                normalized_path = os.path.abspath(os.path.normpath(decoder_model_path))
            except Exception:
                normalized_path = str(decoder_model_path)
            cache_key = f"{normalized_path}#{provider}"

            # Ensure registry caches initialized and cast for static checkers
            decoder_sessions = get_decoder_sessions()

            # First check if session is already cached
            cached_session = decoder_sessions.get(cache_key)
            if cached_session is not None:
                return cached_session

            # Session not in cache - check memory and clear if needed
            from app.utils.cache_manager import CacheManager

            CacheManager.check_and_clear_cache_if_needed()

            def create_session():
                return ort.InferenceSession(decoder_model_path, providers=[provider])

            return decoder_sessions.get_or_add(cache_key, create_session)
        except Exception as e:
            logger.error(f"Failed to create decoder session: {e}")
            return None

    # Public wrappers to avoid external modules accessing protected members directly
    @staticmethod
    def get_model_config(model_to_use: str) -> Any:
        """Public accessor for model configuration (compat).

        Delegates to the internal `_get_model_config` implementation which
        contains the actual loading and caching logic.
        """
        return BaseNLPService._get_model_config(model_to_use)

    @classmethod
    def _get_root_path(cls) -> str:
        """Public accessor for the internal `_root_path` attribute."""
        return cls._root_path

    @classmethod
    def get_root_path(cls) -> str:
        """Backward-compatible public API: return the configured root path."""
        return cls._get_root_path()

    @staticmethod
    def _clear_thread_tokenizers() -> None:
        """Clear thread-local tokenizer cache."""
        cache_clear_thread_tokenizers()

    @staticmethod
    def clear_thread_tokenizers() -> None:
        """Backward-compatible public API to clear thread-local tokenizers."""
        return BaseNLPService._clear_thread_tokenizers()

    @staticmethod
    def _clear_model_config_cache() -> None:
        """Clear model configuration cache."""
        cache_clear_model_config_cache()
        logger.info("Model configuration cache cleared")

    @staticmethod
    def clear_model_config_cache() -> None:
        """Public wrapper for clearing the model config cache (compat)."""
        return BaseNLPService._clear_model_config_cache()

    @staticmethod
    def _get_cache_stats() -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        # Delegate to cache_registry for comprehensive stats
        try:
            stats = cache_get_cache_stats()
            # Add model metadata cache size
            stats["model_metadata_cache"] = BaseNLPService._model_metadata_cache.size()
            return stats
        except Exception:
            # Ensure registry caches initialized and cast before fallback access
            return {
                "encoder_sessions": get_encoder_sessions().size(),
                "model_configs": MODEL_CONFIG_CACHE.size(),
                "model_metadata_cache": BaseNLPService._model_metadata_cache.size(),
            }

    @staticmethod
    def get_cache_stats() -> Dict[str, int]:
        """Backward-compatible public API to fetch cache statistics."""
        return BaseNLPService._get_cache_stats()

    @staticmethod
    def _get_model_metadata(model_to_use: str) -> Optional[dict]:
        """Return lightweight metadata for a model as a plain dict or None.

        Historically callers expected a dict and used `.get()` on the result.
        To preserve backward compatibility we return a `dict` here while the
        cache stores `ModelMetadata` instances internally.
        """
        try:
            val = BaseNLPService._model_metadata_cache.get(model_to_use)
            if val is None:
                return None
            # If stored as a dict, return as-is
            if isinstance(val, dict):
                return val
            # If stored as ModelMetadata, return its dict representation
            if isinstance(val, ModelMetadata):
                return val.model_dump()
            # Unknown stored type: attempt to coerce to dict
            try:
                return dict(val)
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _set_model_metadata(model_to_use: str, metadata: Any) -> None:
        """Store lightweight metadata for a model.

        Accepts either a `ModelMetadata` instance or a plain `dict` for
        backward compatibility. Stored value will be a `ModelMetadata`.
        """
        try:
            if isinstance(metadata, ModelMetadata):
                md = metadata
            elif isinstance(metadata, dict):
                try:
                    md = ModelMetadata(**metadata)
                except Exception:
                    md = ModelMetadata(**dict(metadata))
            else:
                # Try to coerce arbitrary objects
                try:
                    md = ModelMetadata(**dict(metadata))
                except Exception:
                    logger.debug(
                        "Unable to coerce metadata to ModelMetadata for %s",
                        model_to_use,
                    )
                    return

            BaseNLPService._model_metadata_cache.put(model_to_use, md)
        except Exception:
            logger.debug(f"Failed to set model metadata for {model_to_use}")

    @staticmethod
    def _clear_model_metadata() -> None:
        """Clear the model metadata cache."""
        BaseNLPService._model_metadata_cache.clear()

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
                keyword in name for name in output_names for keyword in ["logit", "score", "prob"]
            ):
                return True

        # Shape heuristic - likely vocab size
        arr = outputs[0]
        return arr.ndim >= 2 and arr.shape[-1] <= vocab_threshold

    @staticmethod
    def _warm_up_cache(model_names: List[str]) -> None:
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

    # NOTE: `_get_model_cache_key` compatibility wrapper removed. Call
    # `get_model_cache_key` from `app.utils.cache_keys` directly.

    @staticmethod
    def _get_seq2seq_model(model_to_use: str) -> Optional[Any]:
        """Centralized loader for ORTModelForSeq2SeqLM models.

        Accepts either a model name (resolved via config) or a filesystem
        path / HF repo id. Prefers loading from local path with
        `local_files_only=True` when available. Uses the shared
        `_models` cache and performs memory-based cache eviction via
        `CacheManager` when necessary.
        """
        if ORTModelForSeq2SeqLM is None:
            logger.warning("optimum[onnxruntime] not installed; skipping Seq2SeqLM load")
            return None

        try:
            # Determine canonical cache key for consistent caching behavior
            from app.utils.cache_keys import get_model_cache_key

            cache_key = get_model_cache_key(model_to_use)

            # Ensure registry caches initialized and cast for static checkers
            models_cache = get_models_cache()

            # Return cached model if present
            cached = models_cache.get(cache_key)
            if cached is not None:
                return cached

            # Check memory and clear caches if needed (local import to avoid cycles)
            from app.utils.cache_manager import CacheManager

            CacheManager.check_and_clear_cache_if_needed()

            def _load_impl():
                # When cache_key points to an existing filesystem path, only
                # attempt a local-only load. If that fails, propagate the error
                # rather than trying to interpret the path as a HF repo id.
                if os.path.exists(cache_key):
                    return cast(Any, ORTModelForSeq2SeqLM).from_pretrained(
                        cache_key, use_cache=False, local_files_only=True
                    )

                # Otherwise, allow hub-based loading using the provided identifier
                return cast(Any, ORTModelForSeq2SeqLM).from_pretrained(
                    model_to_use, use_cache=False
                )

            try:
                model = models_cache.get_or_add(cache_key, _load_impl)
            except Exception as e:
                # If the canonical key or the original identifier appears to be
                # a filesystem path (absolute path or existing path), Optimum/HF
                # internals may attempt to validate it as a repo id and raise
                # HFValidationError. In that case, don't propagate the error
                # as a hard failure here; allow callers to fallback to ONNX
                # session-based generation by returning None.
                try:
                    # Only treat the identifier as a path candidate when it is
                    # actually path-like (str/bytes/os.PathLike). This avoids
                    # passing arbitrary objects to `os.path` helpers which can
                    # raise unexpected errors or trigger side effects.
                    if isinstance(cache_key, (str, bytes, os.PathLike)) or isinstance(
                        model_to_use, (str, bytes, os.PathLike)
                    ):
                        is_path_candidate = (
                            os.path.exists(str(cache_key))
                            or os.path.isabs(str(cache_key))
                            or os.path.isabs(str(model_to_use))
                        )
                    else:
                        # Helpful debug: record non-path-like types so we can
                        # diagnose unexpected identifiers passed here in the future.
                        try:
                            logger.debug(
                                "Cache key not path-like: cache_key_type=%s, model_to_use_type=%s",
                                type(cache_key).__name__,
                                type(model_to_use).__name__,
                            )
                        except Exception:
                            # Don't let logging interfere with fallback behavior
                            pass
                        is_path_candidate = False
                except Exception:
                    is_path_candidate = False

                if is_path_candidate:
                    logger.warning(
                        "Failed to load Optimum model from path-like identifier %s: %s. Falling back to ONNX sessions.",
                        sanitize_for_log(cache_key),
                        sanitize_for_log(str(e)),
                    )
                    return None
                raise

            # Compatibility patch
            if model is not None and not hasattr(model, "_supports_cache_class"):
                logger.debug("Applying compatibility patch for ORTModelForSeq2SeqLM")
                model._supports_cache_class = False

            return model
        except FileNotFoundError:
            raise ModelNotFoundError(f"Model not found: {model_to_use}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
