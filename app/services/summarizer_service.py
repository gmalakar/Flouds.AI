# =============================================================================
# File: summarizer_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import asyncio
import json
import os
import time
from typing import Any, Dict, Optional, Set

import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from pydantic import BaseModel, Field

from app.config.onnx_config import OnnxConfig
from app.logger import get_logger
from app.models.summarization_request import (
    SummarizationBatchRequest,
    SummarizationRequest,
)
from app.models.summarization_response import SummarizationResponse
from app.modules.concurrent_dict import ConcurrentDict
from app.services.base_nlp_service import BaseNLPService
from app.utils.batch_limiter import BatchLimiter

logger = get_logger("summarizer_service")


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
            with open(special_tokens_path, "r", encoding="utf-8") as f:
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

            logger.debug(f"Loaded special tokens: {tokens}")
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
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    @staticmethod
    def clear_model_cache():
        """Clear model/session/special token caches (for testing/reloading)."""
        logger.info("Clearing model/session/special token caches.")
        TextSummarizer._models.clear()
        TextSummarizer._decoder_sessions.clear()
        TextSummarizer._special_tokens.clear()

    @staticmethod
    def summarize(request: SummarizationRequest) -> SummarizationResponse:
        """Summarize single text."""
        start_time = time.time()
        logger.debug(
            f"Summarizing text for model: {request.model}, input: {request.input[:100]}"
        )
        response = SummarizationResponse(
            success=True,
            message="Summarization generated successfully",
            model=request.model or "t5-small",
            results=[],
        )

        try:
            import threading
            import time as time_module

            timeout = getattr(request, "timeout", 60)
            timeout_occurred = [False]

            def timeout_handler():
                timeout_occurred[0] = True

            timer = threading.Timer(timeout, timeout_handler)
            timer.start()

            try:
                result = TextSummarizer._summarize_local(request)
                if timeout_occurred[0]:
                    response.success = False
                    response.message = (
                        f"Summarization timed out after {timeout} seconds"
                    )
                    logger.warning(f"Summarization timeout for model: {request.model}")
                elif result.success:
                    response.results.append(result.summary)
                    response.message = result.message
                else:
                    response.success = False
                    response.message = result.message
            finally:
                timer.cancel()
        except Exception as e:
            response.success = False
            response.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during summarization")
        finally:
            response.time_taken = time.time() - start_time
            return response

    @staticmethod
    def _summarize_local(request: SummarizationRequest) -> SummaryResults:
        """Core summarization logic."""
        model_to_use = request.model or "t5-small"
        result = SummaryResults(summary="", message="", success=True)

        try:
            model_config = TextSummarizer._get_model_config(model_to_use)
            if not model_config:
                result.success = False
                result.message = f"Failed to load config for model: {model_to_use}"
                logger.error(result.message)
                return result

            model_to_use_path = os.path.join(
                TextSummarizer._root_path,
                "models",
                model_config.summarization_task or "s2s",
                model_to_use,
            )
            logger.debug(f"Using model path: {model_to_use_path}")

            use_legacy = getattr(model_config, "legacy_tokenizer", False)
            tokenizer = TextSummarizer._get_tokenizer_threadsafe(
                model_to_use_path, use_legacy
            )
            if not tokenizer:
                result.success = False
                result.message = f"Failed to load tokenizer: {model_to_use_path}"
                logger.error(result.message)
                return result

            if getattr(model_config, "use_seq2seqlm", False):
                model = TextSummarizer.get_model(model_to_use_path)
                if not model:
                    result.success = False
                    result.message = (
                        f"Failed to load Seq2SeqLM model: {model_to_use_path}"
                    )
                    logger.error(result.message)
                    return result

                result = TextSummarizer._summarize_seq2seq(
                    model, tokenizer, model_config, request
                )
            else:
                use_optimized = getattr(model_config, "use_optimized", False)
                if use_optimized:
                    encoder_filename = getattr(
                        model_config,
                        "encoder_optimized_onnx_model",
                        "encoder_model_optimized.onnx",
                    )
                    decoder_filename = getattr(
                        model_config,
                        "decoder_optimized_onnx_model",
                        "decoder_model_optimized.onnx",
                    )
                else:
                    encoder_filename = (
                        model_config.encoder_onnx_model or "encoder_model.onnx"
                    )
                    decoder_filename = (
                        model_config.decoder_onnx_model or "decoder_model.onnx"
                    )

                encoder_model_path = os.path.join(model_to_use_path, encoder_filename)
                decoder_model_path = os.path.join(model_to_use_path, decoder_filename)
                logger.debug(
                    f"Encoder path: {encoder_model_path}, Decoder path: {decoder_model_path}"
                )

                encoder_session = TextSummarizer._get_encoder_session(
                    encoder_model_path
                )
                decoder_session = TextSummarizer._get_decoder_session(
                    decoder_model_path
                )

                if not encoder_session or not decoder_session:
                    result.success = False
                    result.message = "Failed to load ONNX sessions"
                    logger.error(result.message)
                    return result

                special_tokens_path = os.path.join(
                    model_to_use_path,
                    model_config.special_tokens_map_path or "special_tokens_map.json",
                )
                special_tokens = TextSummarizer._get_special_tokens(special_tokens_path)

                result = TextSummarizer._summarize_onnx(
                    encoder_session,
                    decoder_session,
                    tokenizer,
                    special_tokens,
                    model_config,
                    request,
                )

        except Exception as e:
            result.success = False
            result.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during summarization")

        return result

    @staticmethod
    def _summarize_seq2seq(
        model: ORTModelForSeq2SeqLM,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: SummarizationRequest,
    ) -> SummaryResults:
        """Summarize using Seq2SeqLM model."""
        result = SummaryResults(summary="", message="", success=True)

        try:
            # Prepare generation parameters
            generate_kwargs = {}

            # Basic parameters
            max_length = getattr(model_config, "max_length", 128)
            if max_length:
                generate_kwargs["max_length"] = max_length

            min_length = getattr(model_config, "min_length", 0)
            if min_length:
                generate_kwargs["min_length"] = min_length

            num_beams = getattr(model_config, "num_beams", 1)
            if num_beams > 1:
                generate_kwargs["num_beams"] = num_beams

            if getattr(model_config, "early_stopping", False):
                generate_kwargs["early_stopping"] = True

            # Additional generation parameters (improvement)
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

            # Temperature handling
            temperature = request.temperature or getattr(
                model_config, "temperature", 0.0
            )
            if temperature > 0.0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["do_sample"] = True

            logger.debug(f"Generation parameters: {generate_kwargs}")

            # Tokenize input
            input_text = TextSummarizer._prepend_text(
                request.input, getattr(model_config, "prepend_text", None)
            )
            logger.debug(f"Input text for summarization: {input_text[:100]}")
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate summary
            summary_ids = model.generate(**inputs, **generate_kwargs)
            logger.debug(f"Generated token IDs: {summary_ids.tolist()}")
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            summary = TextSummarizer._capitalize_sentences(summary)
            result.summary = summary
            if not summary:
                logger.warning(f"Empty summary generated for input: {input_text[:100]}")

        except Exception as e:
            logger.exception("Seq2SeqLM summarization failed")
            result.success = False
            result.message = f"Error generating summarization: {str(e)}"

        return result

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
        result = SummaryResults(summary="", message="", success=True)

        try:
            # Get token IDs from config or tokenizer
            pad_token_id = getattr(
                model_config, "pad_token_id", getattr(tokenizer, "pad_token_id", 0)
            )
            eos_token_id = getattr(
                model_config, "eos_token_id", getattr(tokenizer, "eos_token_id", 1)
            )
            max_length = getattr(model_config, "max_length", 128)

            # Get decoder start token properly
            decoder_start_token_id = getattr(
                model_config, "decoder_start_token_id", None
            )
            if decoder_start_token_id is None:
                # Use tokenizer's decoder start token if available
                if (
                    hasattr(tokenizer, "decoder_start_token_id")
                    and tokenizer.decoder_start_token_id is not None
                ):
                    decoder_start_token_id = tokenizer.decoder_start_token_id
                elif (
                    hasattr(tokenizer, "bos_token_id")
                    and tokenizer.bos_token_id is not None
                ):
                    decoder_start_token_id = tokenizer.bos_token_id
                else:
                    decoder_start_token_id = pad_token_id

            # Tokenize input
            input_text = TextSummarizer._prepend_text(
                request.input, getattr(model_config, "prepend_text", None)
            )
            logger.debug(f"Input text for ONNX summarization: {input_text[:100]}")
            inputs = tokenizer(
                input_text, return_tensors="np", truncation=True, max_length=max_length
            )

            # Prepare encoder inputs with correct data types
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

            # Run encoder
            encoder_outputs = encoder_session.run(None, onnx_inputs)
            logger.debug(
                f"Encoder outputs shapes: {[arr.shape for arr in encoder_outputs]}"
            )

            # Prepare decoder inputs
            decoder_input_names = model_config.decoder_inputnames
            decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)

            # Generate tokens with proper logits handling
            temperature = request.temperature or getattr(
                model_config, "temperature", 0.0
            )
            summary_ids = []

            # Add step timeout for ONNX generation
            step_start = time.time()
            step_timeout = (
                getattr(request, "timeout", 60) / max_length
            )  # Distribute timeout across steps

            for step in range(max_length):
                # Check step timeout
                if time.time() - step_start > step_timeout * (step + 1):
                    logger.warning(f"ONNX generation step timeout at step {step}")
                    break
                decoder_inputs = {
                    getattr(
                        decoder_input_names, "encoder_output", "encoder_hidden_states"
                    ): encoder_outputs[0],
                    getattr(
                        decoder_input_names, "input", "input_ids"
                    ): decoder_input_ids,
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

                # Get logits and apply temperature
                logits_arr = decoder_outputs[0]
                logger.debug(f"Decoder output shape: {logits_arr.shape}")
                if logits_arr.ndim == 3:
                    logits = logits_arr[:, -1, :][0]  # (batch, seq, vocab)
                elif logits_arr.ndim == 2:
                    logits = logits_arr[-1, :]  # (seq, vocab)
                else:
                    logits = logits_arr  # (vocab,)

                if temperature > 0.0:
                    probs = TextSummarizer._softmax(logits / temperature)
                    next_token_id = int(np.random.choice(len(probs), p=probs))
                else:
                    next_token_id = int(np.argmax(logits))

                # Validate token ID
                vocab_size = getattr(tokenizer, "vocab_size", 32000)
                if next_token_id < 0 or next_token_id >= vocab_size:
                    logger.warning(f"Invalid token ID generated: {next_token_id}")
                    break

                summary_ids.append(next_token_id)

                if next_token_id == eos_token_id:
                    logger.debug("EOS token reached")
                    break

                # Update decoder input for next iteration
                decoder_input_ids = np.concatenate(
                    [decoder_input_ids, [[next_token_id]]], axis=1
                )

            # Decode summary
            if summary_ids:
                logger.debug(f"Generated token IDs: {summary_ids}")
                summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
                summary = TextSummarizer._remove_special_tokens(summary, special_tokens)
                summary = TextSummarizer._capitalize_sentences(summary)
                result.summary = summary
                if not summary:
                    logger.warning(
                        f"Empty summary generated for input: {input_text[:100]}"
                    )
            else:
                logger.warning("No valid tokens generated")
                result.summary = ""

        except Exception as e:
            logger.exception("ONNX summarization failed")
            result.success = False
            result.message = f"Error generating summarization: {str(e)}"

        return result

    @staticmethod
    def _remove_special_tokens(text: str, special_tokens: Set[str]) -> str:
        """Remove special tokens from text."""
        import re

        removed = []
        for token in special_tokens:
            if token in text:
                removed.append(token)
                text = text.replace(token, " ")
        if removed:
            logger.debug(f"Removed special tokens: {removed}")
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
            f"Batch summarization for model: {request.model}, batch size: {len(request.inputs)}"
        )
        response = SummarizationResponse(
            success=True,
            message="Batch summarization generated successfully",
            model=request.model,
            results=[],
        )

        try:
            # Validate batch size
            BatchLimiter.validate_batch_size(request.inputs, max_size=20)

            loop = asyncio.get_event_loop()
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

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception) or (result and not result.success):
                    logger.error(
                        f"Error in input {idx}: {getattr(result, 'message', str(result))}"
                    )
                    response.success = False
                    response.message = f"Error in input {idx}: {getattr(result, 'message', str(result))}"
                else:
                    response.results.append(result.summary)

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
        import asyncio

        return asyncio.run(cls.summarize_batch_async(request))
