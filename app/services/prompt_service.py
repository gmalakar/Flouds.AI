# =============================================================================
# File: prompt_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import json
import os
import re
import time
from asyncio import TimeoutError as AsyncTimeoutError
from asyncio import gather, get_event_loop, run
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np
import onnxruntime as ort
from numpy import ndarray

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
except ImportError:
    ORTModelForSeq2SeqLM = None
import hashlib

from pydantic import BaseModel, Field

from app.config.onnx_config import OnnxConfig
from app.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ProcessingTimeoutError,
    TokenizerError,
)
from app.logger import get_logger
from app.models.prompt_request import PromptBatchRequest, PromptRequest
from app.models.prompt_response import PromptResponse
from app.modules.concurrent_dict import ConcurrentDict
from app.services.base_nlp_service import BaseNLPService
from app.services.cache_registry import (
    get_decode_cache,
    get_encoder_output_cache,
    get_generation_cache,
)
from app.utils.batch_limiter import BatchLimiter
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.path_validator import validate_safe_path

logger = get_logger("prompt_service")

# Constants
DEFAULT_MODEL = "t5-small"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 20
DEFAULT_VOCAB_SIZE = 32000
MEMORY_LOW_THRESHOLD_MB = 150  # Clear cache if available memory below this
_CACHE_LIMIT_DECODER = int(os.getenv("FLOUDS_DECODER_CACHE_MAX", "3"))
_CACHE_LIMIT_MODELS = int(os.getenv("FLOUDS_MODEL_CACHE_MAX", "2"))
_CACHE_LIMIT_SPECIAL = int(os.getenv("FLOUDS_SPECIAL_TOKENS_CACHE_MAX", "8"))


class SummaryResults(BaseModel):
    summary: str
    message: str
    success: bool = Field(default=True)


class PromptProcessor(BaseNLPService):
    """Static class for text summarization using ONNX models."""

    _decoder_sessions: ConcurrentDict = ConcurrentDict(
        "_decoder_sessions", max_size=_CACHE_LIMIT_DECODER
    )
    _models: ConcurrentDict = ConcurrentDict("_models", max_size=_CACHE_LIMIT_MODELS)
    _special_tokens: ConcurrentDict = ConcurrentDict(
        "_special_tokens", max_size=_CACHE_LIMIT_SPECIAL
    )

    @staticmethod
    def _load_special_tokens(special_tokens_path: str) -> Set[str]:
        """Load special tokens from JSON file."""
        logger.debug(
            "Loading special tokens from: %s", sanitize_for_log(special_tokens_path)
        )
        if not os.path.exists(special_tokens_path):
            logger.warning(f"Special tokens file not found: {special_tokens_path}")
            return set()

        try:
            from app.utils.path_validator import safe_open

            with safe_open(
                special_tokens_path, PromptProcessor._root_path, "r", encoding="utf-8"
            ) as f:
                data = json.load(f)

            tokens = set()
            for key in ["pad_token", "eos_token", "unk_token"]:
                if key in data and "content" in data[key]:
                    tokens.add(data[key]["content"])

            if "additional_special_tokens" in data:
                if isinstance(data["additional_special_tokens"], list):
                    tokens.update(data["additional_special_tokens"])
                elif isinstance(data["additional_special_tokens"], dict):
                    tokens.update(data["additional_special_tokens"].values())
            return tokens
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in special tokens file: {e}")
            return set()
        except Exception as e:
            logger.error(f"Failed to load special tokens: {e}")
            return set()

    @staticmethod
    def _get_vocab_size_from_session(session: ort.InferenceSession) -> Optional[int]:
        """Extract vocabulary size from ONNX session output shape (for language models)."""
        try:
            outputs = session.get_outputs()
            if outputs:
                for output in outputs:
                    # Look for logits output with shape [batch, seq_len, vocab_size]
                    if output.shape and len(output.shape) >= 3:
                        vocab_dim = output.shape[-1]
                        if (
                            isinstance(vocab_dim, (int, np.integer))
                            and vocab_dim > 1000
                        ):
                            # Reasonable vocab size (> 1000 tokens)
                            logger.debug(
                                f"Auto-detected vocab_size from ONNX output: {vocab_dim}"
                            )
                            return int(vocab_dim)
        except Exception as e:
            logger.warning(f"Could not auto-detect vocab_size from ONNX session: {e}")
        return None

    @staticmethod
    def _get_decoder_session(decoder_model_path: str) -> Optional[ort.InferenceSession]:
        """Get cached ONNX decoder session with error handling and provider support."""
        try:
            from app.app_init import APP_SETTINGS

            provider = APP_SETTINGS.server.session_provider or "CPUExecutionProvider"
            available_providers = ort.get_available_providers()
            if provider not in available_providers:
                logger.warning(
                    f"Provider {provider} not available for decoder, using CPUExecutionProvider"
                )
                provider = "CPUExecutionProvider"
            cache_key = f"{decoder_model_path}#{provider}"

            # First check if session is already cached
            cached_session = PromptProcessor._decoder_sessions.get(cache_key)
            if cached_session is not None:
                return cached_session

            # Session not in cache - check memory and clear if needed
            from app.utils.cache_manager import CacheManager

            CacheManager.check_and_clear_cache_if_needed()

            logger.debug(
                "Creating decoder session for %s with provider %s",
                sanitize_for_log(decoder_model_path),
                sanitize_for_log(provider),
            )
            return PromptProcessor._decoder_sessions.get_or_add(
                cache_key,
                lambda: ort.InferenceSession(decoder_model_path, providers=[provider]),
            )
        except Exception as e:
            logger.error(f"Failed to create decoder session: {e}")
            return None

    @staticmethod
    def _get_special_tokens(special_tokens_path: str) -> Set[str]:
        """Get cached special tokens."""
        return PromptProcessor._special_tokens.get_or_add(
            special_tokens_path,
            lambda: PromptProcessor._load_special_tokens(special_tokens_path),
        )

    @staticmethod
    def get_model(model_to_use_path: str) -> Optional[Any]:
        """Get cached ONNX model with error handling."""
        if ORTModelForSeq2SeqLM is None:
            logger.warning(
                "optimum[onnxruntime] not installed; skipping Seq2SeqLM load"
            )
            return None

        try:
            # First check if model is already cached
            cached_model = PromptProcessor._models.get(model_to_use_path)
            if cached_model is not None:
                return cached_model

            # Model not in cache - check memory and clear if needed
            from app.utils.cache_manager import CacheManager

            CacheManager.check_and_clear_cache_if_needed()

            logger.debug(f"Loading model from path: {model_to_use_path}")
            model = PromptProcessor._models.get_or_add(
                model_to_use_path,
                # Cast to Any to avoid static analyzer complaining when ORTModelForSeq2SeqLM
                # may be None at import-time; runtime check above prevents reaching this.
                lambda: cast(Any, ORTModelForSeq2SeqLM).from_pretrained(
                    model_to_use_path, use_cache=False
                ),
            )
            # Compatibility patch for newer transformers versions
            if not hasattr(model, "_supports_cache_class"):
                logger.debug("Applying compatibility patch for ORTModelForSeq2SeqLM")
                model._supports_cache_class = False

            logger.info(f"Model cache size: {PromptProcessor._models.size()}")
            return model
        except FileNotFoundError:
            raise ModelNotFoundError(f"Model not found: {model_to_use_path}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    @staticmethod
    def clear_model_cache():
        """Clear model/session/special token caches (for testing/reloading)."""
        logger.info("Clearing model/session/special token caches.")
        PromptProcessor._models.clear()
        PromptProcessor._decoder_sessions.clear()
        PromptProcessor._special_tokens.clear()

    @staticmethod
    def _check_memory_and_clear_cache() -> None:
        """Check available memory and clear caches if below threshold."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            threshold_bytes = MEMORY_LOW_THRESHOLD_MB * 1024 * 1024
            if mem.available < threshold_bytes:
                logger.warning(
                    f"Low memory before generation: {mem.available / 1024 / 1024:.1f}MB available"
                )
                from app.utils.cache_manager import CacheManager

                CacheManager.clear_all_caches()
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")

    @staticmethod
    def process_prompt(request: PromptRequest) -> PromptResponse:
        """Process prompt for text generation/summarization."""
        start_time = time.time()
        logger.debug(
            "Processing prompt for model: %s, input: %s",
            sanitize_for_log(request.model),
            sanitize_for_log(request.input[:100]),
        )
        response = PromptResponse(
            success=True,
            message="Prompt processed successfully",
            model=request.model or DEFAULT_MODEL,
            results=[],
            time_taken=0.0,
        )

        try:
            import threading

            timeout = getattr(request, "timeout", DEFAULT_TIMEOUT)
            timeout_occurred = [False]

            def timeout_handler():
                timeout_occurred[0] = True

            timer = threading.Timer(timeout, timeout_handler)
            timer.start()

            try:
                result = PromptProcessor._process_prompt_local(request)
                if timeout_occurred[0]:
                    raise ProcessingTimeoutError(
                        f"Prompt processing timed out after {timeout} seconds"
                    )
                elif result.success:
                    response.results.append(result.summary)
                    response.message = result.message
                else:
                    response.success = False
                    response.message = result.message
            finally:
                timer.cancel()
        finally:
            response.time_taken = time.time() - start_time
            return response

    @staticmethod
    def summarize(request: PromptRequest) -> PromptResponse:
        """Backward compatibility method for summarization."""
        return PromptProcessor.process_prompt(request)

    @staticmethod
    def _process_prompt_local(request: PromptRequest) -> SummaryResults:
        """Core prompt processing logic."""
        model_to_use = request.model or DEFAULT_MODEL

        try:
            model_config = PromptProcessor._get_model_config(model_to_use)
            if not model_config:
                return SummaryResults(
                    summary="",
                    message=f"Model '{model_to_use}' not found",
                    success=False,
                )

            model_path, tokenizer = PromptProcessor._prepare_model_resources(
                model_config, model_to_use
            )

            # Record which generation path is selected for diagnosis.
            try:
                use_seq2seqlm = bool(getattr(model_config, "use_seq2seqlm", False))
                encoder_only = bool(getattr(model_config, "encoder_only", False))
                logger.info(
                    "Generation path selection for model=%s: use_seq2seqlm=%s encoder_only=%s",
                    sanitize_for_log(model_to_use),
                    use_seq2seqlm,
                    encoder_only,
                )
            except Exception:
                logger.debug("Failed to log generation path flags", exc_info=True)

            if use_seq2seqlm:
                logger.info(
                    "Using seq2seq generation for %s", sanitize_for_log(model_to_use)
                )
                return PromptProcessor._run_seq2seq_generation(
                    model_path, tokenizer, model_config, request
                )
            elif encoder_only:
                logger.info(
                    "Using encoder-only generation for %s",
                    sanitize_for_log(model_to_use),
                )
                return PromptProcessor._run_encoder_only_generation(
                    model_path, tokenizer, model_config, request
                )
            else:
                logger.info(
                    "Using encoder+decoder ONNX generation for %s",
                    sanitize_for_log(model_to_use),
                )
                return PromptProcessor._run_onnx_generation(
                    model_path, tokenizer, model_config, request
                )

        except FileNotFoundError:
            raise ModelNotFoundError("Model files not accessible")
        except OSError as e:
            raise ModelLoadError(f"System error accessing model: {e}")
        except (ValueError, KeyError) as e:
            raise InferenceError(f"Invalid model configuration: {e}")
        except Exception as e:
            raise InferenceError(f"Error generating text: {e}")

    @staticmethod
    def _prepare_model_resources(
        model_config: OnnxConfig, model_name: str
    ) -> tuple[str, Any]:
        """Prepare model path and tokenizer."""
        # Resolve model folder first (tests may patch this) and then validate
        # availability using the supplied model_config so we don't re-fetch
        # via the centralized getter. Skip filesystem checks here because
        # tokenizer/session loading will perform strict validation.
        model_path = PromptProcessor._get_model_path(model_name)
        if not model_path:
            raise ModelLoadError(f"Failed to resolve model path for {model_name}")
        if not BaseNLPService._validate_model_availability(
            model_name,
            task="prompt",
            perform_filesystem_check=False,
            cfg=model_config,
            model_path=model_path,
        ):
            raise ModelLoadError(
                f"Model '{model_name}' not available or model files missing"
            )

        logger.info("Using model path: %s", sanitize_for_log(model_path))

        use_legacy = getattr(model_config, "legacy_tokenizer", False)
        tokenizer = PromptProcessor._get_tokenizer_threadsafe(model_path, use_legacy)
        if not tokenizer:
            raise TokenizerError(f"Failed to load tokenizer: {model_path}")

        return model_path, tokenizer

    @staticmethod
    def _run_seq2seq_generation(
        model_path: str,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: PromptRequest,
    ) -> SummaryResults:
        """Run Seq2SeqLM text generation."""
        if ORTModelForSeq2SeqLM is None:
            logger.info("Seq2SeqLM backend unavailable; falling back to ONNX sessions")
            return PromptProcessor._run_onnx_generation(
                model_path, tokenizer, model_config, request
            )

        model = PromptProcessor.get_model(model_path)
        if not model:
            logger.info(
                "Seq2SeqLM model missing; falling back to ONNX sessions for %s",
                sanitize_for_log(model_path),
            )
            return PromptProcessor._run_onnx_generation(
                model_path, tokenizer, model_config, request
            )

        return PromptProcessor._generate_seq2seq(
            model, tokenizer, model_config, request
        )

    @staticmethod
    def _run_encoder_only_generation(
        model_path: str,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: PromptRequest,
    ) -> SummaryResults:
        """Run encoder-only ONNX text generation for GPT-style models."""
        encoder_filename = PromptProcessor._get_model_filename(model_config, "encoder")
        encoder_path = validate_safe_path(
            os.path.join(model_path, encoder_filename), PromptProcessor._root_path
        )

        encoder_session = PromptProcessor._get_encoder_session(encoder_path)
        if not encoder_session:
            raise ModelLoadError("Failed to load encoder session")

        # Auto-detect vocab_size from encoder session if not in config
        if not hasattr(model_config, "vocab_size") or model_config.vocab_size is None:
            vocab_size = PromptProcessor._get_vocab_size_from_session(encoder_session)
            if vocab_size:
                model_config.vocab_size = vocab_size
                logger.info(f"Auto-detected vocab_size from ONNX model: {vocab_size}")

        return PromptProcessor._generate_encoder_only(
            encoder_session, tokenizer, model_config, request
        )

    @staticmethod
    def _run_onnx_generation(
        model_path: str,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: PromptRequest,
    ) -> SummaryResults:
        """Run ONNX encoder/decoder text generation."""
        encoder_path, decoder_path = PromptProcessor._get_model_file_paths(
            model_path, model_config
        )

        encoder_session = PromptProcessor._get_encoder_session(encoder_path)
        decoder_session = PromptProcessor._get_decoder_session(decoder_path)

        if not encoder_session or not decoder_session:
            raise ModelLoadError("Failed to load ONNX sessions")

        # Auto-detect vocab_size from decoder session if not in config
        if not hasattr(model_config, "vocab_size") or model_config.vocab_size is None:
            vocab_size = PromptProcessor._get_vocab_size_from_session(decoder_session)
            if vocab_size:
                model_config.vocab_size = vocab_size
                logger.info(f"Auto-detected vocab_size from ONNX decoder: {vocab_size}")

        special_tokens_path = validate_safe_path(
            os.path.join(
                model_path,
                model_config.special_tokens_map_path or "special_tokens_map.json",
            ),
            PromptProcessor._root_path,
        )
        special_tokens = PromptProcessor._get_special_tokens(special_tokens_path)

        return PromptProcessor._generate_onnx(
            encoder_session,
            decoder_session,
            tokenizer,
            special_tokens,
            model_config,
            request,
        )

    @staticmethod
    def _get_model_file_paths(
        model_path: str, model_config: OnnxConfig
    ) -> tuple[str, str]:
        """Get encoder and decoder model file paths."""
        encoder_filename = PromptProcessor._get_model_filename(model_config, "encoder")
        decoder_filename = PromptProcessor._get_model_filename(model_config, "decoder")

        encoder_path = validate_safe_path(
            os.path.join(model_path, encoder_filename), PromptProcessor._root_path
        )
        decoder_path = validate_safe_path(
            os.path.join(model_path, decoder_filename), PromptProcessor._root_path
        )

        logger.debug(f"Encoder path: {encoder_path}, Decoder path: {decoder_path}")
        return encoder_path, decoder_path

    @staticmethod
    def _get_model_filename(model_config: OnnxConfig, model_type: str) -> str:
        """Get model filename based on type and optimization settings."""
        use_optimized = getattr(model_config, "use_optimized", False)

        if model_type == "encoder":
            if use_optimized:
                return getattr(
                    model_config,
                    "encoder_optimized_onnx_model",
                    "encoder_model_optimized.onnx",
                )
            else:
                return model_config.encoder_onnx_model or "encoder_model.onnx"
        else:  # decoder
            if use_optimized:
                return getattr(
                    model_config,
                    "decoder_optimized_onnx_model",
                    "decoder_model_optimized.onnx",
                )
            else:
                return model_config.decoder_onnx_model or "decoder_model.onnx"

    @staticmethod
    def _generate_seq2seq(
        model: Any,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: PromptRequest,
    ) -> SummaryResults:
        """Generate text using Seq2SeqLM model."""
        # Preemptive memory check
        PromptProcessor._check_memory_and_clear_cache()

        try:
            generate_kwargs = PromptProcessor._build_generation_params(
                model_config, request
            )

            # Some ONNX models loaded via optimum do not support PKV cache reuse.
            # When calling `model.generate` with an Optimum ORT model, force
            # `use_cache=False` to avoid the runtime ValueError described below.
            try:
                mod = getattr(model.__class__, "__module__", "") or ""
                if "optimum" in mod or mod.startswith("optimum"):
                    generate_kwargs.setdefault("use_cache", False)
            except Exception:
                # Best-effort; don't fail generation parameter building on inspection
                pass
            input_text = PromptProcessor._prepare_input_text(request, model_config)
            inputs = tokenizer(input_text, return_tensors="pt")

            # Use GenerationConfig to avoid deprecation warning
            try:
                from transformers import GenerationConfig

                generation_config = GenerationConfig(**generate_kwargs)
                summary_ids = model.generate(
                    **inputs, generation_config=generation_config
                )
            except ImportError:
                # Fallback for older transformers versions
                summary_ids = model.generate(**inputs, **generate_kwargs)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            summary = PromptProcessor._capitalize_sentences(summary)

            if not summary:
                logger.warning(f"Empty output generated for input: {input_text[:100]}")

            return SummaryResults(summary=summary, message="", success=True)

        except MemoryError as e:
            logger.error("Out of memory during Seq2Seq generation")
            try:
                from app.utils.cache_manager import CacheManager

                CacheManager.clear_all_caches()
            except:
                pass
            return SummaryResults(
                summary="",
                message="Out of memory: input text too large. Please reduce size and try again.",
                success=False,
            )
        except Exception as e:
            logger.exception("Seq2SeqLM generation failed")
            return SummaryResults(
                summary="",
                message=f"Error generating text: {str(e)}",
                success=False,
            )

    @staticmethod
    def _build_generation_params(
        model_config: OnnxConfig, request: PromptRequest
    ) -> Dict[str, Any]:
        """Build generation parameters for model."""
        generate_kwargs = {}

        # Basic parameters
        PromptProcessor._add_basic_params(generate_kwargs, model_config)
        PromptProcessor._add_beam_params(generate_kwargs, model_config)
        PromptProcessor._add_optional_params(generate_kwargs, model_config)
        PromptProcessor._add_temperature_params(generate_kwargs, model_config, request)

        logger.debug(f"Generation parameters: {generate_kwargs}")
        return generate_kwargs

    @staticmethod
    def _add_basic_params(
        generate_kwargs: Dict[str, Any], model_config: OnnxConfig
    ) -> None:
        """Add basic generation parameters."""
        for param, default in [("max_length", DEFAULT_MAX_LENGTH), ("min_length", 0)]:
            value = getattr(model_config, param, default)
            if value is not None:
                generate_kwargs[param] = value

    @staticmethod
    def _add_beam_params(
        generate_kwargs: Dict[str, Any], model_config: OnnxConfig
    ) -> None:
        """Add beam search parameters."""
        num_beams = getattr(model_config, "num_beams", 1)
        if num_beams > 1:
            generate_kwargs["num_beams"] = num_beams
        if getattr(model_config, "early_stopping", False):
            generate_kwargs["early_stopping"] = True

    @staticmethod
    def _add_optional_params(
        generate_kwargs: Dict[str, Any], model_config: OnnxConfig
    ) -> None:
        """Add optional generation parameters."""
        for param in [
            "repetition_penalty",
            "length_penalty",
            "top_k",
            "top_p",
            "no_repeat_ngram_size",
        ]:
            value = getattr(model_config, param, None)
            if value is not None:
                generate_kwargs[param] = value

    @staticmethod
    def _add_temperature_params(
        generate_kwargs: Dict[str, Any],
        model_config: OnnxConfig,
        request: PromptRequest,
    ) -> None:
        """Add temperature and sampling parameters."""
        temperature = request.temperature or getattr(model_config, "temperature", 0.0)
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True

    @staticmethod
    def _prepare_input_text(request: PromptRequest, model_config: OnnxConfig) -> str:
        """Prepare input text for summarization."""
        input_text = PromptProcessor._prepend_text(
            request.input, getattr(model_config, "prepend_text", None)
        )
        logger.debug(
            "Input text for summarization: %s",
            sanitize_for_log(str(input_text[:100])),
        )
        return input_text

    @staticmethod
    def _generate_onnx(
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        tokenizer: Any,
        special_tokens: Set[str],
        model_config: OnnxConfig,
        request: PromptRequest,
    ) -> SummaryResults:
        """Generate text using ONNX encoder/decoder sessions."""
        # Preemptive memory check to avoid OOM crashes
        PromptProcessor._check_memory_and_clear_cache()

        try:
            token_config = PromptProcessor._get_token_config(model_config, tokenizer)
            input_text = PromptProcessor._prepare_input_text(request, model_config)

            # Tokenize and run encoder
            inputs = tokenizer(
                input_text,
                return_tensors="np",
                truncation=True,
                max_length=token_config["max_length"],
            )
            encoder_outputs = PromptProcessor._run_encoder(
                encoder_session, inputs, model_config
            )

            # Generate tokens
            summary_ids = PromptProcessor._generate_tokens(
                decoder_session,
                encoder_outputs,
                inputs,
                token_config,
                model_config,
                request,
            )

            # Decode and process output
            summary = PromptProcessor._decode_output(
                summary_ids, tokenizer, special_tokens, input_text
            )
            return SummaryResults(summary=summary, message="", success=True)

        except MemoryError as e:
            logger.error("Out of memory during ONNX generation")
            # Trigger aggressive cleanup
            try:
                from app.utils.cache_manager import CacheManager

                CacheManager.clear_all_caches()
            except:
                pass
            return SummaryResults(
                summary="",
                message="Out of memory: input text too large. Please reduce size and try again.",
                success=False,
            )
        except Exception as e:
            logger.exception("ONNX generation failed")
            return SummaryResults(
                summary="",
                message=f"Error generating text: {str(e)}",
                success=False,
            )

    @staticmethod
    def _get_token_config(model_config: OnnxConfig, tokenizer: Any) -> Dict[str, int]:
        """Get token configuration for ONNX inference."""
        pad_token_id = getattr(
            model_config, "pad_token_id", getattr(tokenizer, "pad_token_id", 0)
        )
        eos_token_id = getattr(
            model_config, "eos_token_id", getattr(tokenizer, "eos_token_id", 1)
        )
        max_length = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)

        # Get decoder start token
        decoder_start_token_id = (
            getattr(model_config, "decoder_start_token_id", None)
            or getattr(tokenizer, "decoder_start_token_id", None)
            or getattr(tokenizer, "bos_token_id", None)
            or pad_token_id
        )

        return {
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "decoder_start_token_id": decoder_start_token_id,
            "max_length": max_length,
        }

    @staticmethod
    def _run_encoder(
        encoder_session: ort.InferenceSession, inputs: Dict, model_config: OnnxConfig
    ) -> Any:
        """Run encoder inference."""
        input_names = model_config.inputnames
        # Normalize input arrays to numpy ndarrays before casting
        try:
            in_ids = np.ascontiguousarray(np.asarray(inputs["input_ids"]))
            in_ids = in_ids.astype(np.int64)
        except Exception:
            in_ids = inputs["input_ids"]

        onnx_inputs = {getattr(input_names, "input", "input_ids"): in_ids}

        if "attention_mask" in inputs:
            try:
                mask = np.ascontiguousarray(np.asarray(inputs["attention_mask"]))
                mask = mask.astype(np.int64)
            except Exception:
                mask = inputs["attention_mask"]
            onnx_inputs[getattr(input_names, "mask", "attention_mask")] = mask

        result = encoder_session.run(None, onnx_inputs)
        return result

    @staticmethod
    def _hash_array(arr: np.ndarray, quantize_dtype: Any = np.float16) -> str:
        """Hash a numpy array after optional quantization.

        Returns a hex digest string. Centralized to avoid duplication in cache
        key builders and tests.
        """
        try:
            a = np.ascontiguousarray(arr.astype(quantize_dtype))
        except Exception:
            a = np.ascontiguousarray(np.asarray(arr))
        h = hashlib.sha256()
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        try:
            h.update(memoryview(a))
        except Exception:
            h.update(a.tobytes())
        return h.hexdigest()

    @staticmethod
    def _build_generation_cache_key(
        decoder_session: ort.InferenceSession,
        encoder_outputs: Any,
        inputs: Dict,
        token_config: Dict[str, int],
        model_config: OnnxConfig,
        request: PromptRequest,
        tokenizer: Optional[Any] = None,
    ) -> Optional[str]:
        """Build generation cache key including hashed inputs and encoder output."""
        try:
            model_id = getattr(request, "model", "") or ""
            model_version = getattr(model_config, "model_folder_name", "") or ""
            tokenizer_id = None
            if tokenizer is not None:
                tokenizer_id = getattr(tokenizer, "name_or_path", None) or getattr(
                    tokenizer, "model_max_length", None
                )

            parts = {
                "model": model_id,
                "model_version": model_version or "",
                "tokenizer": tokenizer_id or "",
                "max_length": int(token_config.get("max_length", 0)),
                "decoder_start_token_id": int(
                    token_config.get("decoder_start_token_id", 0)
                ),
                "eos_token_id": int(token_config.get("eos_token_id", 0)),
                "tenant": getattr(request, "tenant_code", None) or "",
            }

            m = hashlib.sha256()
            m.update(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode())

            if isinstance(inputs, dict) and "input_ids" in inputs:
                ids = np.ascontiguousarray(inputs["input_ids"]).astype(np.int64)
                m.update(b"||input_ids||")
                m.update(str(ids.shape).encode())
                m.update(PromptProcessor._hash_array(ids, np.int64).encode())

            if isinstance(inputs, dict) and "attention_mask" in inputs:
                mask = np.ascontiguousarray(inputs["attention_mask"]).astype(np.int8)
                m.update(b"||attention_mask||")
                m.update(str(mask.shape).encode())
                m.update(PromptProcessor._hash_array(mask, np.int8).encode())

            try:
                if (
                    isinstance(encoder_outputs, (list, tuple))
                    and len(encoder_outputs) > 0
                ):
                    enc0 = np.ascontiguousarray(encoder_outputs[0])
                    enc_digest = PromptProcessor._hash_array(enc0, np.float16)
                    m.update(b"||enc0||")
                    m.update(str(enc0.shape).encode())
                    m.update(enc_digest.encode())
            except Exception:
                m.update(b"||enc_hash_failed||")

            return m.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _generate_tokens(
        decoder_session: ort.InferenceSession,
        encoder_outputs: Any,
        inputs: Dict,
        token_config: Dict[str, int],
        model_config: OnnxConfig,
        request: PromptRequest,
        tokenizer: Optional[Any] = None,
    ) -> List[int]:
        """Generate tokens using decoder."""
        decoder_input_names = model_config.decoder_inputnames
        decoder_input_ids = np.array(
            [[token_config["decoder_start_token_id"]]], dtype=np.int64
        )
        temperature = request.temperature or getattr(model_config, "temperature", 0.0)
        summary_ids = []

        # Attempt to use centralized deterministic generation cache
        try:
            gen_cache = get_generation_cache()
        except Exception:
            gen_cache = None

        cache_key = None
        if gen_cache is not None:
            try:
                cache_key = PromptProcessor._build_generation_cache_key(
                    decoder_session,
                    encoder_outputs,
                    inputs,
                    token_config,
                    model_config,
                    request,
                    tokenizer,
                )
                if cache_key:
                    cached = gen_cache.get(cache_key)
                    if cached is not None:
                        # Return a shallow copy to prevent caller-side mutation
                        return list(cached)
            except Exception:
                pass

        step_start = time.time()
        step_timeout = (
            getattr(request, "timeout", DEFAULT_TIMEOUT) / token_config["max_length"]
        )

        # Ensure encoder_outputs[0] is a numpy array for the decoder inputs
        try:
            enc0 = np.ascontiguousarray(np.asarray(encoder_outputs[0]))
        except Exception:
            enc0 = encoder_outputs[0]

        for step in range(token_config["max_length"]):
            if time.time() - step_start > step_timeout * (step + 1):
                logger.warning(f"ONNX generation step timeout at step {step}")
                break

            decoder_inputs = {
                getattr(
                    decoder_input_names, "encoder_output", "encoder_hidden_states"
                ): enc0,
                getattr(decoder_input_names, "input", "input_ids"): decoder_input_ids,
            }

            if "attention_mask" in inputs:
                try:
                    mask_arr = np.ascontiguousarray(
                        np.asarray(inputs["attention_mask"])
                    )
                    decoder_inputs[
                        getattr(decoder_input_names, "mask", "encoder_attention_mask")
                    ] = mask_arr.astype(np.int64)
                except Exception:
                    decoder_inputs[
                        getattr(decoder_input_names, "mask", "encoder_attention_mask")
                    ] = inputs["attention_mask"]

            try:
                decoder_outputs = decoder_session.run(None, decoder_inputs)
            except Exception as e:
                logger.error(f"Decoder inference error: {e}")
                break

            # Normalize decoder output to numpy array for safe indexing
            try:
                dec_out0 = np.ascontiguousarray(np.asarray(decoder_outputs[0]))
            except Exception:
                dec_out0 = decoder_outputs[0]

            # Get advanced sampling parameters from config
            top_k = getattr(model_config, "top_k", None)
            top_p = getattr(model_config, "top_p", None)
            repetition_penalty = getattr(model_config, "repetition_penalty", None)

            next_token_id = PromptProcessor._sample_next_token(
                dec_out0,
                temperature,
                top_k,
                top_p,
                summary_ids,
                repetition_penalty,
            )

            # Validate token ID
            vocab_size = getattr(model_config, "vocab_size", DEFAULT_VOCAB_SIZE)
            if next_token_id < 0 or next_token_id >= vocab_size:
                logger.warning(f"Invalid token ID generated: {next_token_id}")
                break

            summary_ids.append(next_token_id)

            if next_token_id == token_config["eos_token_id"]:
                logger.debug("EOS token reached")
                break

            decoder_input_ids = np.concatenate(
                [decoder_input_ids, [[next_token_id]]], axis=1
            )

        return summary_ids

    @staticmethod
    def _apply_repetition_penalty(
        logits: np.ndarray,
        generated_ids: Optional[list],
        repetition_penalty: Optional[float],
    ) -> np.ndarray:
        """Apply repetition penalty to logits and return modified array."""
        try:
            if (
                repetition_penalty is None
                or repetition_penalty == 1.0
                or not generated_ids
            ):
                return logits
            out = logits.copy()
            for token_id in set(generated_ids):
                if 0 <= token_id < out.shape[0]:
                    if out[token_id] < 0:
                        out[token_id] *= repetition_penalty
                    else:
                        out[token_id] /= repetition_penalty
            return out
        except Exception:
            return logits

    @staticmethod
    def _filter_top_k(logits: np.ndarray, top_k: Optional[int]) -> np.ndarray:
        """Keep only top_k logits, set others to -inf."""
        try:
            if top_k is None or top_k <= 0:
                return logits
            top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
            filtered = np.full_like(logits, -np.inf)
            filtered[top_k_indices] = logits[top_k_indices]
            return filtered
        except Exception:
            return logits

    @staticmethod
    def _filter_top_p(probs: np.ndarray, top_p: Optional[float]) -> np.ndarray:
        """Apply nucleus (top-p) filtering to probability vector and renormalize."""
        try:
            if top_p is None or top_p >= 1.0:
                return probs
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1
            cutoff_idx = min(cutoff_idx, len(sorted_indices))
            filtered = np.zeros_like(probs)
            filtered[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
            if filtered.sum() > 0:
                return filtered / filtered.sum()
            return probs
        except Exception:
            return probs

    @staticmethod
    def _sample_from_probs(probs: np.ndarray) -> int:
        try:
            return int(np.random.choice(len(probs), p=probs))
        except Exception:
            return int(np.argmax(probs))

    @staticmethod
    def _sample_next_token(
        logits_arr: Any,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generated_ids: Optional[list] = None,
        repetition_penalty: Optional[float] = None,
    ) -> int:
        """Sample next token from logits with top_k, top_p, and repetition penalty.

        Delegates sub-steps to small helpers to keep this function concise
        for the unit test that enforces function length limits.
        """
        # Coerce any array-like / tensor-like input to a numpy ndarray
        arr = np.asarray(logits_arr)
        if arr.ndim == 3:
            logits = arr[0, -1, :]
        elif arr.ndim == 2:
            logits = arr[-1, :]
        else:
            logits = arr

        logits = PromptProcessor._apply_repetition_penalty(
            logits, generated_ids, repetition_penalty
        )

        if temperature <= 0.0:
            return int(np.argmax(logits))

        logits = logits / temperature
        logits = PromptProcessor._filter_top_k(logits, top_k)
        probs = PromptProcessor._softmax(logits)
        probs = PromptProcessor._filter_top_p(probs, top_p)
        return PromptProcessor._sample_from_probs(probs)

    @staticmethod
    def _decode_output(
        output_ids: List[int],
        tokenizer: Any,
        special_tokens: Set[str],
        input_text: str,
    ) -> str:
        """Decode and process generated output."""
        if not output_ids:
            logger.warning("No valid tokens generated")
            return ""

        # Attempt to use a decode cache keyed by tokenizer identity, skip flag,
        # token ids and special_tokens set. This prevents repeated expensive
        # decoding + post-processing for identical outputs.
        try:
            dec_cache = get_decode_cache()
        except Exception:
            dec_cache = None

        cache_key = None
        if dec_cache is not None:
            try:
                tokenizer_id = getattr(tokenizer, "name_or_path", None) or getattr(
                    tokenizer, "model_max_length", None
                )
                parts = {
                    "tokenizer": str(tokenizer_id) or "",
                    "skip_special": True,
                    "ids": list(map(int, output_ids)),
                    "special_tokens": (
                        sorted([t for t in special_tokens]) if special_tokens else []
                    ),
                }
                m = hashlib.sha256()
                m.update(
                    json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
                )
                cache_key = m.hexdigest()
                cached = dec_cache.get(cache_key)
                if cached is not None:
                    return cached
            except Exception:
                cache_key = None

        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        output = PromptProcessor._remove_special_tokens(output, special_tokens)
        output = PromptProcessor._capitalize_sentences(output)

        if not output:
            logger.warning(f"Empty output generated for input: {input_text[:100]}")

        try:
            if cache_key and dec_cache is not None:
                dec_cache.put(cache_key, output)
        except Exception:
            pass

        return output

    @staticmethod
    def _generate_encoder_only(
        encoder_session: ort.InferenceSession,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: PromptRequest,
    ) -> SummaryResults:
        """Generate text using encoder-only model (GPT-style)."""
        try:
            input_text = request.input

            # Add BOS token if specified in config
            bos_token_id = getattr(model_config, "bos_token_id", None)
            if bos_token_id is not None:
                # Tokenize without special tokens first
                inputs = tokenizer(
                    input_text,
                    return_tensors="np",
                    truncation=True,
                    max_length=getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)
                    - 1,
                    add_special_tokens=False,
                )
                # Prepend BOS token
                input_ids = inputs["input_ids"]
                input_ids = np.concatenate(
                    [np.array([[bos_token_id]]), input_ids], axis=1
                )
                inputs["input_ids"] = input_ids
                if "attention_mask" in inputs:
                    attention_mask = np.concatenate(
                        [np.array([[1]]), inputs["attention_mask"]], axis=1
                    )
                    inputs["attention_mask"] = attention_mask
            else:
                inputs = tokenizer(
                    input_text,
                    return_tensors="np",
                    truncation=True,
                    max_length=getattr(model_config, "max_length", DEFAULT_MAX_LENGTH),
                )

            # Normalize tokenizer outputs to numpy arrays to avoid editor/type warnings
            try:
                inputs["input_ids"] = np.ascontiguousarray(
                    np.asarray(inputs["input_ids"])
                )
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = np.ascontiguousarray(
                        np.asarray(inputs["attention_mask"])
                    )
            except Exception:
                # Best-effort normalization; fall back to original values on failure
                pass

            # Run encoder inference
            try:
                enc_in_ids = np.ascontiguousarray(np.asarray(inputs["input_ids"]))
                enc_in_ids = enc_in_ids.astype(np.int64)
            except Exception:
                enc_in_ids = inputs["input_ids"]

            onnx_inputs = {"input_ids": enc_in_ids}
            if "attention_mask" in inputs:
                try:
                    enc_mask = np.ascontiguousarray(
                        np.asarray(inputs["attention_mask"])
                    )
                    enc_mask = enc_mask.astype(np.int64)
                except Exception:
                    enc_mask = inputs["attention_mask"]
                onnx_inputs["attention_mask"] = enc_mask

            # Add position_ids if required by the model
            seq_len = inputs["input_ids"].shape[1]
            onnx_inputs["position_ids"] = np.arange(seq_len, dtype=np.int64)[None, :]

            outputs = encoder_session.run(None, onnx_inputs)
            # Ensure logits is a contiguous numpy array for safe indexing
            try:
                logits = np.ascontiguousarray(np.asarray(outputs[0]))
            except Exception:
                logits = outputs[0]

            # Generate next tokens
            temperature = request.temperature or getattr(
                model_config, "temperature", 0.8
            )
            top_k = getattr(model_config, "top_k", None)
            top_p = getattr(model_config, "top_p", None)
            repetition_penalty = getattr(model_config, "repetition_penalty", None)
            max_new_tokens = (
                getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)
                - inputs["input_ids"].shape[1]
            )
            # Defensive cap: never allow extremely large generation loops.
            if max_new_tokens is None or max_new_tokens <= 0:
                max_new_tokens = 0
            else:
                max_new_tokens = int(max_new_tokens)
            max_new_tokens = min(max_new_tokens, DEFAULT_MAX_LENGTH)

            # Time-based safety: avoid long-running generation when timeout supplied
            step_start = time.time()
            step_timeout = getattr(request, "timeout", DEFAULT_TIMEOUT) / max(
                1, max_new_tokens
            )

            generated_ids = inputs["input_ids"][0].tolist()

            for step in range(max_new_tokens):
                # Break if per-step timeout exceeded (defensive guard)
                try:
                    if time.time() - step_start > step_timeout * (step + 1):
                        logger.warning(
                            f"Encoder-only generation step timeout at step {step}"
                        )
                        break
                except Exception:
                    pass
                # Pass the full logits array; the sampler will handle slicing
                next_token_id = PromptProcessor._sample_next_token(
                    logits,
                    temperature,
                    top_k,
                    top_p,
                    generated_ids,
                    repetition_penalty,
                )
                generated_ids.append(next_token_id)

                # Check for EOS token
                if next_token_id == getattr(model_config, "eos_token_id", 50256):
                    break

                # Prepare next input
                new_input = np.array([generated_ids], dtype=np.int64)
                onnx_inputs["input_ids"] = new_input
                if "attention_mask" in onnx_inputs:
                    onnx_inputs["attention_mask"] = np.ones_like(
                        new_input, dtype=np.int64
                    )
                # Update position_ids for new sequence length
                new_seq_len = new_input.shape[1]
                onnx_inputs["position_ids"] = np.arange(new_seq_len, dtype=np.int64)[
                    None, :
                ]

                outputs = encoder_session.run(None, onnx_inputs)
                try:
                    logits = np.ascontiguousarray(np.asarray(outputs[0]))
                except Exception:
                    logits = outputs[0]

            # Decode output (skip input tokens)
            output_ids = generated_ids[inputs["input_ids"].shape[1] :]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            return SummaryResults(summary=output_text, message="", success=True)

        except Exception as e:
            logger.exception("Encoder-only generation failed")
            return SummaryResults(
                summary="",
                message=f"Error generating text: {str(e)}",
                success=False,
            )

    @staticmethod
    def _remove_special_tokens(text: str, special_tokens: Set[str]) -> str:
        """Remove special tokens from text using regex for batch removal."""
        if not special_tokens or not text:
            return text

        # Escape special regex characters and create pattern
        escaped_tokens = [re.escape(token) for token in special_tokens if token]
        if not escaped_tokens:
            return text

        pattern = "|".join(escaped_tokens)
        original_text = text
        text = re.sub(pattern, " ", text)

        if text != original_text:
            logger.debug(f"Removed special tokens using pattern: {pattern}")

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        """Capitalize the first word of each sentence."""
        if not text:
            return text
        sentences = re.split(r"([.!?]\s*)", text)
        result = []
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part.strip():
                part = part.strip()
                if part:
                    part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
            result.append(part)
        return "".join(result).strip()

    @staticmethod
    async def summarize_batch_async(
        request: PromptBatchRequest,
    ) -> PromptResponse:
        """Asynchronous batch summarization with partial success reporting."""
        start_time = time.time()
        logger.debug(
            "Batch summarization for model: %s, batch size: %d",
            sanitize_for_log(request.model),
            len(request.inputs),
        )
        response = PromptResponse(
            success=True,
            message="Batch prompt processing completed successfully",
            model=request.model or DEFAULT_MODEL,
            results=[],
            time_taken=0.0,
        )

        try:
            # Validate batch size
            BatchLimiter.validate_batch_size(
                request.inputs, max_size=DEFAULT_BATCH_SIZE
            )

            loop = get_event_loop()
            try:
                tasks = [
                    loop.run_in_executor(
                        None,
                        PromptProcessor._process_prompt_local,
                        PromptRequest(
                            model=request.model,
                            input=text,
                            temperature=getattr(request, "temperature", None),
                            tenant_code=request.tenant_code,
                        ),
                    )
                    for text in request.inputs
                ]

                results = await gather(*tasks, return_exceptions=True)
            finally:
                # Don't close the running event loop - it's managed by FastAPI
                pass
            for idx, result in enumerate(results):
                # Handle exceptions and objects without `success` safely
                if isinstance(result, Exception) or (
                    hasattr(result, "success") and not getattr(result, "success")
                ):
                    err_msg = getattr(result, "message", str(result))
                    logger.error(f"Error in input {idx}: {err_msg}")
                    response.success = False
                    response.message = f"Error in input {idx}: {err_msg}"
                else:
                    summary_val = getattr(result, "summary", None)
                    if summary_val is not None:
                        response.results.append(summary_val)

        except (ValueError, TypeError) as e:
            response.success = False
            response.message = "Invalid batch request parameters"
            logger.error("Batch validation error: %s", str(e))
        except (AsyncTimeoutError, TimeoutError) as e:
            response.success = False
            response.message = "Batch summarization timed out"
            logger.error("Batch timeout: %s", str(e))
        except Exception as e:
            response.success = False
            response.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during batch summarization")
        finally:
            response.time_taken = time.time() - start_time
            return response

    @classmethod
    def summarize_batch(cls, request):
        """Backward compatibility for tests."""
        return run(cls.summarize_batch_async(request))
