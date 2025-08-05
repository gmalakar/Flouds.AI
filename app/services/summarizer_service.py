# =============================================================================
# File: summarizer_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import json
import os
import time
from asyncio import TimeoutError as AsyncTimeoutError
from asyncio import gather, get_event_loop, run
from typing import Any, Dict, Optional, Set

import numpy as np
import onnxruntime as ort
from numpy import ndarray
from optimum.onnxruntime import ORTModelForSeq2SeqLM
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
from app.models.summarization_request import (
    SummarizationBatchRequest,
    SummarizationRequest,
)
from app.models.summarization_response import SummarizationResponse
from app.modules.concurrent_dict import ConcurrentDict
from app.services.base_nlp_service import BaseNLPService
from app.utils.batch_limiter import BatchLimiter
from app.utils.error_handler import ErrorHandler, handle_errors
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.path_validator import validate_safe_path

logger = get_logger("summarizer_service")

# Constants
DEFAULT_MODEL = "t5-small"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 20
DEFAULT_VOCAB_SIZE = 32000


class SummaryResults(BaseModel):
    summary: str
    message: str
    success: bool = Field(default=True)


class TextSummarizer(BaseNLPService):
    """Static class for text summarization using ONNX models."""

    _decoder_sessions: ConcurrentDict = ConcurrentDict("_decoder_sessions")
    _models: ConcurrentDict = ConcurrentDict("_models")
    _special_tokens: ConcurrentDict = ConcurrentDict("_special_tokens")

    @staticmethod
    def _load_special_tokens(special_tokens_path: str) -> Set[str]:
        """Load special tokens from JSON file."""
        logger.debug(f"Loading special tokens from: {special_tokens_path}")
        if not os.path.exists(special_tokens_path):
            logger.warning(f"Special tokens file not found: {special_tokens_path}")
            return set()

        try:
            from app.utils.path_validator import safe_open

            with safe_open(
                special_tokens_path, TextSummarizer._root_path, "r", encoding="utf-8"
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
            logger.debug(
                f"Creating decoder session for {decoder_model_path} with provider {provider}"
            )
            return TextSummarizer._decoder_sessions.get_or_add(
                cache_key,
                lambda: ort.InferenceSession(decoder_model_path, providers=[provider]),
            )
        except Exception as e:
            logger.error(f"Failed to create decoder session: {e}")
            return None

    @staticmethod
    def _get_special_tokens(special_tokens_path: str) -> Set[str]:
        """Get cached special tokens."""
        return TextSummarizer._special_tokens.get_or_add(
            special_tokens_path,
            lambda: TextSummarizer._load_special_tokens(special_tokens_path),
        )

    @staticmethod
    def get_model(model_to_use_path: str) -> Optional[ORTModelForSeq2SeqLM]:
        """Get cached ONNX model with error handling."""
        try:
            logger.debug(f"Loading model from path: {model_to_use_path}")
            return TextSummarizer._models.get_or_add(
                model_to_use_path,
                lambda: ORTModelForSeq2SeqLM.from_pretrained(
                    model_to_use_path, use_cache=False
                ),
            )
        except FileNotFoundError:
            raise ModelNotFoundError(f"Model not found: {model_to_use_path}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    @staticmethod
    def clear_model_cache():
        """Clear model/session/special token caches (for testing/reloading)."""
        logger.info("Clearing model/session/special token caches.")
        TextSummarizer._models.clear()
        TextSummarizer._decoder_sessions.clear()
        TextSummarizer._special_tokens.clear()

    @staticmethod
    @handle_errors(context="summarization", include_traceback=True)
    def summarize(request: SummarizationRequest) -> SummarizationResponse:
        """Summarize single text."""
        start_time = time.time()
        logger.debug(
            "Summarizing text for model: %s, input: %s",
            sanitize_for_log(request.model),
            sanitize_for_log(request.input[:100]),
        )
        response = SummarizationResponse(
            success=True,
            message="Summarization generated successfully",
            model=request.model or DEFAULT_MODEL,
            results=[],
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
                result = TextSummarizer._summarize_local(request)
                if timeout_occurred[0]:
                    raise ProcessingTimeoutError(
                        f"Summarization timed out after {timeout} seconds"
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
    def _summarize_local(request: SummarizationRequest) -> SummaryResults:
        """Core summarization logic."""
        model_to_use = request.model or DEFAULT_MODEL

        try:
            model_config = TextSummarizer._get_model_config(model_to_use)
            if not model_config:
                raise ModelLoadError(f"Failed to load config for model: {model_to_use}")

            model_path, tokenizer = TextSummarizer._prepare_model_resources(
                model_config, model_to_use
            )

            if getattr(model_config, "use_seq2seqlm", False):
                return TextSummarizer._run_seq2seq_summarization(
                    model_path, tokenizer, model_config, request
                )
            else:
                return TextSummarizer._run_onnx_summarization(
                    model_path, tokenizer, model_config, request
                )

        except FileNotFoundError:
            raise ModelNotFoundError("Model files not accessible")
        except OSError as e:
            raise ModelLoadError(f"System error accessing model: {e}")
        except (ValueError, KeyError) as e:
            raise InferenceError(f"Invalid model configuration: {e}")
        except Exception as e:
            raise InferenceError(f"Error generating summarization: {e}")

    @staticmethod
    def _prepare_model_resources(
        model_config: OnnxConfig, model_name: str
    ) -> tuple[str, Any]:
        """Prepare model path and tokenizer."""
        model_path = validate_safe_path(
            os.path.join(
                TextSummarizer._root_path,
                "models",
                model_config.summarization_task or "s2s",
                model_name,
            ),
            TextSummarizer._root_path,
        )
        logger.debug(f"Using model path: {model_path}")

        use_legacy = getattr(model_config, "legacy_tokenizer", False)
        tokenizer = TextSummarizer._get_tokenizer_threadsafe(model_path, use_legacy)
        if not tokenizer:
            raise TokenizerError(f"Failed to load tokenizer: {model_path}")

        return model_path, tokenizer

    @staticmethod
    def _run_seq2seq_summarization(
        model_path: str,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: SummarizationRequest,
    ) -> SummaryResults:
        """Run Seq2SeqLM summarization."""
        model = TextSummarizer.get_model(model_path)
        if not model:
            raise ModelLoadError(f"Failed to load Seq2SeqLM model: {model_path}")

        return TextSummarizer._summarize_seq2seq(
            model, tokenizer, model_config, request
        )

    @staticmethod
    def _run_onnx_summarization(
        model_path: str,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: SummarizationRequest,
    ) -> SummaryResults:
        """Run ONNX encoder/decoder summarization."""
        encoder_path, decoder_path = TextSummarizer._get_model_file_paths(
            model_path, model_config
        )

        encoder_session = TextSummarizer._get_encoder_session(encoder_path)
        decoder_session = TextSummarizer._get_decoder_session(decoder_path)

        if not encoder_session or not decoder_session:
            raise ModelLoadError("Failed to load ONNX sessions")

        special_tokens_path = validate_safe_path(
            os.path.join(
                model_path,
                model_config.special_tokens_map_path or "special_tokens_map.json",
            ),
            TextSummarizer._root_path,
        )
        special_tokens = TextSummarizer._get_special_tokens(special_tokens_path)

        return TextSummarizer._summarize_onnx(
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
        encoder_filename = TextSummarizer._get_model_filename(model_config, "encoder")
        decoder_filename = TextSummarizer._get_model_filename(model_config, "decoder")

        encoder_path = validate_safe_path(
            os.path.join(model_path, encoder_filename), TextSummarizer._root_path
        )
        decoder_path = validate_safe_path(
            os.path.join(model_path, decoder_filename), TextSummarizer._root_path
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
    def _summarize_seq2seq(
        model: ORTModelForSeq2SeqLM,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: SummarizationRequest,
    ) -> SummaryResults:
        """Summarize using Seq2SeqLM model."""
        try:
            generate_kwargs = TextSummarizer._build_generation_params(
                model_config, request
            )
            input_text = TextSummarizer._prepare_input_text(request, model_config)
            inputs = tokenizer(input_text, return_tensors="pt")

            summary_ids = model.generate(**inputs, **generate_kwargs)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            summary = TextSummarizer._capitalize_sentences(summary)

            if not summary:
                logger.warning(f"Empty summary generated for input: {input_text[:100]}")

            return SummaryResults(summary=summary, message="", success=True)

        except Exception as e:
            logger.exception("Seq2SeqLM summarization failed")
            return SummaryResults(
                summary="",
                message=f"Error generating summarization: {str(e)}",
                success=False,
            )

    @staticmethod
    def _build_generation_params(
        model_config: OnnxConfig, request: SummarizationRequest
    ) -> Dict[str, Any]:
        """Build generation parameters for model."""
        generate_kwargs = {}

        # Basic parameters
        TextSummarizer._add_basic_params(generate_kwargs, model_config)
        TextSummarizer._add_beam_params(generate_kwargs, model_config)
        TextSummarizer._add_optional_params(generate_kwargs, model_config)
        TextSummarizer._add_temperature_params(generate_kwargs, model_config, request)

        logger.debug(f"Generation parameters: {generate_kwargs}")
        return generate_kwargs

    @staticmethod
    def _add_basic_params(
        generate_kwargs: Dict[str, Any], model_config: OnnxConfig
    ) -> None:
        """Add basic generation parameters."""
        for param, default in [("max_length", DEFAULT_MAX_LENGTH), ("min_length", 0)]:
            value = getattr(model_config, param, default)
            if value:
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
        request: SummarizationRequest,
    ) -> None:
        """Add temperature and sampling parameters."""
        temperature = request.temperature or getattr(model_config, "temperature", 0.0)
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True

    @staticmethod
    def _prepare_input_text(
        request: SummarizationRequest, model_config: OnnxConfig
    ) -> str:
        """Prepare input text for summarization."""
        input_text = TextSummarizer._prepend_text(
            request.input, getattr(model_config, "prepend_text", None)
        )
        logger.debug(
            "Input text for summarization: %s",
            sanitize_for_log(str(input_text[:100])),
        )
        return input_text

    @staticmethod
    def _summarize_onnx(
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        tokenizer: Any,
        special_tokens: Set[str],
        model_config: OnnxConfig,
        request: SummarizationRequest,
    ) -> SummaryResults:
        """Summarize using ONNX encoder/decoder sessions."""
        try:
            token_config = TextSummarizer._get_token_config(model_config, tokenizer)
            input_text = TextSummarizer._prepare_input_text(request, model_config)

            # Tokenize and run encoder
            inputs = tokenizer(
                input_text,
                return_tensors="np",
                truncation=True,
                max_length=token_config["max_length"],
            )
            encoder_outputs = TextSummarizer._run_encoder(
                encoder_session, inputs, model_config
            )

            # Generate tokens
            summary_ids = TextSummarizer._generate_tokens(
                decoder_session,
                encoder_outputs,
                inputs,
                token_config,
                model_config,
                request,
            )

            # Decode and process summary
            summary = TextSummarizer._decode_summary(
                summary_ids, tokenizer, special_tokens, input_text
            )
            return SummaryResults(summary=summary, message="", success=True)

        except Exception as e:
            logger.exception("ONNX summarization failed")
            return SummaryResults(
                summary="",
                message=f"Error generating summarization: {str(e)}",
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
        onnx_inputs = {
            getattr(input_names, "input", "input_ids"): inputs["input_ids"].astype(
                np.int64
            )
        }

        if "attention_mask" in inputs:
            onnx_inputs[getattr(input_names, "mask", "attention_mask")] = inputs[
                "attention_mask"
            ].astype(np.int64)

        return encoder_session.run(None, onnx_inputs)

    @staticmethod
    def _generate_tokens(
        decoder_session: ort.InferenceSession,
        encoder_outputs: Any,
        inputs: Dict,
        token_config: Dict[str, int],
        model_config: OnnxConfig,
        request: SummarizationRequest,
    ) -> list[int]:
        """Generate tokens using decoder."""
        decoder_input_names = model_config.decoder_inputnames
        decoder_input_ids = np.array(
            [[token_config["decoder_start_token_id"]]], dtype=np.int64
        )
        temperature = request.temperature or getattr(model_config, "temperature", 0.0)
        summary_ids = []

        step_start = time.time()
        step_timeout = (
            getattr(request, "timeout", DEFAULT_TIMEOUT) / token_config["max_length"]
        )

        for step in range(token_config["max_length"]):
            if time.time() - step_start > step_timeout * (step + 1):
                logger.warning(f"ONNX generation step timeout at step {step}")
                break

            decoder_inputs = {
                getattr(
                    decoder_input_names, "encoder_output", "encoder_hidden_states"
                ): encoder_outputs[0],
                getattr(decoder_input_names, "input", "input_ids"): decoder_input_ids,
            }

            if "attention_mask" in inputs:
                decoder_inputs[
                    getattr(decoder_input_names, "mask", "encoder_attention_mask")
                ] = inputs["attention_mask"].astype(np.int64)

            try:
                decoder_outputs = decoder_session.run(None, decoder_inputs)
            except Exception as e:
                logger.error(f"Decoder inference error: {e}")
                break

            next_token_id = TextSummarizer._sample_next_token(
                decoder_outputs[0], temperature
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
    def _sample_next_token(logits_arr: ndarray, temperature: float) -> int:
        """Sample next token from logits."""
        if logits_arr.ndim == 3:
            logits = logits_arr[:, -1, :][0]
        elif logits_arr.ndim == 2:
            logits = logits_arr[-1, :]
        else:
            logits = logits_arr

        if temperature > 0.0:
            probs = TextSummarizer._softmax(logits / temperature)
            return int(np.random.choice(len(probs), p=probs))
        else:
            return int(np.argmax(logits))

    @staticmethod
    def _decode_summary(
        summary_ids: list[int],
        tokenizer: Any,
        special_tokens: Set[str],
        input_text: str,
    ) -> str:
        """Decode and process summary."""
        if not summary_ids:
            logger.warning("No valid tokens generated")
            return ""

        summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
        summary = TextSummarizer._remove_special_tokens(summary, special_tokens)
        summary = TextSummarizer._capitalize_sentences(summary)

        if not summary:
            logger.warning(f"Empty summary generated for input: {input_text[:100]}")

        return summary

    @staticmethod
    def _remove_special_tokens(text: str, special_tokens: Set[str]) -> str:
        """Remove special tokens from text using regex for batch removal."""
        import re

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
        import re

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
        request: SummarizationBatchRequest,
    ) -> SummarizationResponse:
        """Asynchronous batch summarization with partial success reporting."""
        start_time = time.time()
        logger.debug(
            "Batch summarization for model: %s, batch size: %d",
            sanitize_for_log(request.model),
            len(request.inputs),
        )
        response = SummarizationResponse(
            success=True,
            message="Batch summarization generated successfully",
            model=request.model,
            results=[],
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
                        TextSummarizer._summarize_local,
                        SummarizationRequest(
                            model=request.model,
                            input=text,
                            temperature=getattr(request, "temperature", None),
                        ),
                    )
                    for text in request.inputs
                ]

                results = await gather(*tasks, return_exceptions=True)
            finally:
                # Don't close the running event loop - it's managed by FastAPI
                pass

            for idx, result in enumerate(results):
                if isinstance(result, Exception) or (result and not result.success):
                    logger.error(
                        f"Error in input {idx}: {getattr(result, 'message', str(result))}"
                    )
                    response.success = False
                    response.message = f"Error in input {idx}: {getattr(result, 'message', str(result))}"
                else:
                    response.results.append(result.summary)

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
