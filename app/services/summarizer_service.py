# =============================================================================
# File: summarizer_service.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""
summarizer_service.py

Provides the TextSummarizer class for ONNX and HuggingFace-based text summarization.
Includes thread-safe caching, dynamic config loading, and robust logging.
"""

import asyncio
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from pydantic import BaseModel, Field
from scipy.special import softmax
from transformers import AutoConfig, GenerationConfig

from app.config.config_loader import ConfigLoader
from app.config.onnx_config import OnnxConfig
from app.logger import get_logger
from app.models.summarization_request import (
    SummarizationBatchRequest,
    SummarizationRequest,
)
from app.models.summarization_response import SummarizationResponse
from app.modules.concurrent_dict import ConcurrentDict
from app.modules.utilities import Utilities
from app.services.base_nlp_service import BaseNLPService
from app.setup import APP_SETTINGS

logger = get_logger("summarizer_service")


class SummaryResults(BaseModel):
    summary: str
    message: str
    success: bool = Field(default=True)


class TextSummarizer(BaseNLPService):
    """
    Static class for text summarization using ONNX or HuggingFace models.
    """

    _MERGED_WITH_RAW_KEY: str = "_merged_with_raw"
    _MISSING_DECODER_START_TOKEN_ID_KEY: str = "_missing_decoder_start_token_id"

    _root_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "onnx")
    )
    _encoder_sessions: ConcurrentDict = ConcurrentDict("_encoder_sessions")
    _decoder_sessions: ConcurrentDict = ConcurrentDict("_decoder_sessions")
    _models: ConcurrentDict = ConcurrentDict("_models")
    _tokenizers: ConcurrentDict = ConcurrentDict("_tokenizers")
    _special_tokens: ConcurrentDict = ConcurrentDict("_special_tokens")
    _auto_configs: ConcurrentDict = ConcurrentDict("_auto_configs")
    _generation_configs: ConcurrentDict = ConcurrentDict("_generation_configs")
    _raw_generation_configs: ConcurrentDict = ConcurrentDict("_raw_generation_configs")
    _generate_kwargs: ConcurrentDict = ConcurrentDict("_generate_kwargs")

    @staticmethod
    def _load_special_tokens(special_tokens_path: str) -> Set[str]:
        """
        Loads special tokens from a JSON file.
        """
        if not os.path.exists(special_tokens_path):
            return set()
        with open(special_tokens_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokens = set()
        for key in ["pad_token", "eos_token", "unk_token"]:
            if key in data and "content" in data[key]:
                tokens.add(data[key]["content"])
        if "additional_special_tokens" in data:
            tokens.update(data["additional_special_tokens"])
        return tokens

    @staticmethod
    def _load_raw_generation_config_dict(model_to_use_path: str) -> Dict[str, Any]:
        config_path = os.path.join(model_to_use_path, "generation_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _get_auto_config(tokenizer_path: str) -> Any:
        """
        Returns a cached auto config for the given vocab path.
        """
        return TextSummarizer._auto_configs.get_or_add(
            tokenizer_path, lambda: AutoConfig.from_pretrained(tokenizer_path)
        )

    @staticmethod
    def _get_generation_config(tokenizer_path: str) -> Dict[str, Any]:
        """
        Returns a cached generation config for the given vocab path.
        """
        return TextSummarizer._generation_configs.get_or_add(
            tokenizer_path,
            lambda: (GenerationConfig.from_pretrained(tokenizer_path) or {}).to_dict(),
        )

    @staticmethod
    def _get_decoder_session(decoder_model_path: str) -> ort.InferenceSession:
        """
        Returns a cached ONNX decoder session.
        """
        return TextSummarizer._decoder_sessions.get_or_add(
            decoder_model_path, lambda: ort.InferenceSession(decoder_model_path)
        )

    @staticmethod
    def _get_special_tokens(special_tokens_path: str) -> Set[str]:
        """
        Returns cached special tokens for the given path.
        """
        return TextSummarizer._special_tokens.get_or_add(
            special_tokens_path,
            lambda: TextSummarizer._load_special_tokens(special_tokens_path),
        )

    @staticmethod
    def _get_raw_generation_config(model_to_use_path: str) -> Dict[str, Any]:
        """
        Returns cached generation config for the given path.
        """
        return TextSummarizer._raw_generation_configs.get_or_add(
            model_to_use_path,
            lambda: TextSummarizer._load_raw_generation_config_dict(model_to_use_path),
        )

    @staticmethod
    def get_model(model_to_use_path: str) -> ORTModelForSeq2SeqLM:
        """
        Returns a cached ONNX model for the given path.
        """
        return TextSummarizer._models.get_or_add(
            model_to_use_path,
            lambda: ORTModelForSeq2SeqLM.from_pretrained(
                model_to_use_path, use_cache=False
            ),
        )

    @classmethod
    def summarize(
        cls,
        request: SummarizationRequest,
    ) -> SummarizationResponse:
        """
        Summarize the input text using the specified model.
        Returns the summary string, or None if summarization fails.
        """
        model_to_use = request.model or "t5-small"

        response = SummarizationResponse(
            success=True,
            message="Summarization generated successfully",
            model=model_to_use,
            results=[],
        )
        start_time = time.time()

        try:
            summaryResults = TextSummarizer._summarize_local(request)
            if summaryResults.success:
                response.results.append(summaryResults.summary)
                response.message = summaryResults.message
            else:
                response.success = False
                response.message = summaryResults.message
        except Exception as e:
            response.success = False
            response.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during summarization")
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"Summarization completed in {elapsed:.2f} seconds.")
            response.time_taken = elapsed
            return response

    @staticmethod
    def _summarize_local(
        request: SummarizationRequest,
    ) -> SummaryResults:
        """
        Summarize the input text using the specified model.
        Returns the summary string, or None if summarization fails.
        """
        model_to_use = request.model or "t5-small"

        summaryResults: SummaryResults = SummaryResults(
            summary="", message="", success=True
        )

        try:
            model_config = TextSummarizer._get_model_config(model_to_use)
            logger.debug(
                f"Summarizing text using model: {model_to_use} and task: {model_config.summarization_task}..."
            )
            model_to_use_path = os.path.join(
                TextSummarizer._root_path,
                "models",
                model_config.summarization_task or "s2s",
                model_to_use,
            )
            encoder_model_path = os.path.join(
                model_to_use_path, model_config.encoder_onnx_model or "model.onnx"
            )
            decoder_model_path = os.path.join(
                model_to_use_path, model_config.decoder_onnx_model or "model.onnx"
            )
            special_tokens_path = os.path.join(
                model_to_use_path,
                model_config.special_tokens_map_path or "special_tokens_map.json",
            )

            tokenizer = TextSummarizer._get_tokenizer_threadsafe(model_to_use_path)
            special_tokens = TextSummarizer._get_special_tokens(special_tokens_path)

            # Allow override of config parameters
            max_length = getattr(model_config, "max_length", 128) or 128

            if model_config.use_seq2seqlm:
                logger.debug(f"Using Seq2SeqLM model: {model_to_use_path}")
                model = TextSummarizer.get_model(model_to_use_path)
                summaryResults = TextSummarizer._summarizeSeq2SeqLm(
                    model, tokenizer, model_config, request, model_to_use_path
                )
            else:
                logger.debug(f"Using encoder/decoder model: {model_to_use_path}")
                encoder_session = TextSummarizer._get_encoder_session(
                    encoder_model_path
                )
                decoder_session = TextSummarizer._get_decoder_session(
                    decoder_model_path
                )
                summaryResults = TextSummarizer._summarizeOther(
                    encoder_session,
                    decoder_session,
                    tokenizer,
                    special_tokens,
                    model_config,
                    request,
                    model_to_use_path,
                )
        except Exception as e:
            summaryResults.success = False
            summaryResults.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during summarization")
        finally:
            logger.debug(f"Summarization completed successfully.")
            return summaryResults

    @classmethod
    async def summarize_batch_async(
        cls, request: SummarizationBatchRequest
    ) -> SummarizationResponse:
        """
        Asynchronously summarize a batch of texts using run_in_executor for concurrency.
        If any summarization fails, stop and return the error.
        """
        start_time = time.time()
        responses = SummarizationResponse(
            success=True,
            message="Batch summarization generated successfully",
            model=request.model,
            results=[],
        )
        try:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    None,
                    TextSummarizer._summarize_local,
                    SummarizationRequest(
                        model=request.model,
                        input=text,
                    ),
                )
                for text in request.inputs
            ]
            results = await asyncio.gather(*tasks)
            for result in results:
                if result.success:
                    responses.results.append(result.summary)
                else:
                    responses.success = False
                    responses.message = result.message
                    break  # Stop adding further results if any failed
        except Exception as e:
            responses.success = False
            responses.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during summarization")
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"Summarization completed in {elapsed:.2f} seconds.")
            responses.time_taken = elapsed
            return responses

    @classmethod
    def summarize_batch(
        cls, request: SummarizationBatchRequest
    ) -> SummarizationResponse:
        """
        Summarize a batch of texts.
        """
        start_time = time.time()
        responses = SummarizationResponse(
            success=True,
            message="Batch summarization generated successfully",
            model=request.model,
            results=[],
        )
        try:
            for text in request.inputs:
                req = SummarizationRequest(
                    model=request.model,
                    input=text,
                )
                response = TextSummarizer._summarize_local(req)
                if response.success:
                    responses.results.append(response.summary)
                else:
                    responses.success = False
                    responses.message = response.message
                    break  # <-- This breaks the loop on failure
        except Exception as e:
            responses.success = False
            responses.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during summarization")
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"Summarization completed in {elapsed:.2f} seconds.")
            responses.time_taken = elapsed
            return responses

    @staticmethod
    def _remove_special_tokens(text: str, special_tokens: Set[str]) -> str:
        """
        Removes special tokens from the text.
        """
        for token in special_tokens:
            text = text.replace(token, "")
        return text.strip()

    @staticmethod
    def _get_key_value_from_generation_dict(
        key: str,
        model_config: OnnxConfig,
        tokenizer_path: str,
        use_model_config_if_available: bool = False,
        model_type: Optional[str] = None,
    ) -> Any:
        """
        Sets generation config parameters based on the text and special tokens.
        """
        gen_config_dict = TextSummarizer._get_generation_config(tokenizer_path)

        if use_model_config_if_available and hasattr(model_config, key):
            logger.debug(f"Using {key} from model config: {getattr(model_config, key)}")
            val = getattr(model_config, key)
            gen_config_dict[key] = val
            return val
        if key not in gen_config_dict or gen_config_dict[key] is None:
            logger.debug(
                f"{key} not found in generation config, checking raw generation config"
            )
            raw_gen_config = TextSummarizer._get_raw_generation_config(tokenizer_path)
            logger.debug(f"Using raw generation config: {raw_gen_config}")
            if key in raw_gen_config:
                logger.debug(
                    f"Using {key} from raw generation config: {raw_gen_config[key]}"
                )
                gen_config_dict[key] = raw_gen_config[key]
            else:
                if key == "decoder_start_token_id":  # handling bert/t5 specific cases
                    logger.debug(
                        f"{key} not found in raw generation config, using pad_token_id"
                    )
                    gen_config_dict[key] = (
                        getattr(
                            TextSummarizer._get_auto_config(tokenizer_path),
                            "decoder_start_token_id",
                            None,
                        )
                        or getattr(model_config, "decoder_start_token_id", None)
                        or TextSummarizer._MISSING_DECODER_START_TOKEN_ID_KEY
                    )
                else:
                    gen_config_dict[key] = (
                        getattr(
                            TextSummarizer._get_auto_config(tokenizer_path), key, None
                        )
                        or getattr(model_config, key, None)
                        or 0
                        if key == "pad_token_id"
                        else (
                            2
                            if key == "eos_token_id"
                            else 128 if key == "max_length" else None
                        )
                    )
        return gen_config_dict[key] or None

    @staticmethod
    def _summarizeSeq2SeqLm(
        model: ORTModelForSeq2SeqLM,
        tokenizer: Any,
        model_config: OnnxConfig,
        request: SummarizationRequest,
        tokenizer_path: str,
    ) -> SummaryResults:
        """
        Summarize using a HuggingFace Seq2SeqLM model.
        """
        summaryResults: SummaryResults = SummaryResults(
            summary="", message="", success=True
        )
        try:
            generation_config = TextSummarizer._get_generation_config(tokenizer_path)
            # Only merge if not already merged
            if generation_config is not None and not generation_config.get(
                TextSummarizer._MERGED_WITH_RAW_KEY, False
            ):
                generation_config = Utilities.add_missing_from_other(
                    generation_config,
                    TextSummarizer._get_raw_generation_config(tokenizer_path),
                )
                generation_config[TextSummarizer._MERGED_WITH_RAW_KEY] = (
                    True  # Mark as merged
                )
                logger.debug(
                    f"Merged generation config with raw config for: {tokenizer_path}"
                )

            if generation_config is not None:
                # Update all keys from the loaded generation_config onto the model's generation_config
                logger.debug(f"Using generation config from: {tokenizer_path}")
                for key, value in generation_config.items():
                    setattr(model.generation_config, key, value)
            num_beams = TextSummarizer._get_key_value_from_generation_dict(
                "num_beams", model_config, tokenizer_path, True
            )
            early_stopping = (
                TextSummarizer._get_key_value_from_generation_dict(
                    "early_stopping", model_config, tokenizer_path, True
                )
                or False
            )
            min_length = (
                TextSummarizer._get_key_value_from_generation_dict(
                    "min_length", model_config, tokenizer_path, True
                )
                or False
            )

            max_length = (
                TextSummarizer._get_key_value_from_generation_dict(
                    "max_length", model_config, tokenizer_path, True
                )
                or 128
            )

            use_generation_config = getattr(
                model_config, "use_generation_config", False
            )

            generate_kwargs = TextSummarizer._generate_kwargs.get_or_add(
                tokenizer_path,
                lambda: {},
            )
            if len(generate_kwargs) == 0:
                if max_length is not None and max_length > 0:
                    logger.debug(f"Using max_length: {max_length} for summarization")
                    generate_kwargs["max_length"] = max_length

                if min_length is not None and min_length > 0:
                    logger.debug(f"Using min_length: {min_length} for summarization")
                    generate_kwargs["min_length"] = min_length

                if (
                    max_length is not None
                    and max_length > 0
                    and early_stopping is not None
                    and early_stopping
                ):
                    logger.debug(
                        f"Using early_stopping: {early_stopping} for summarization"
                    )
                    generate_kwargs["early_stopping"] = early_stopping

                if num_beams is not None and num_beams > 0:
                    logger.debug(f"Using num_beams: {num_beams} for summarization")
                    generate_kwargs["num_beams"] = num_beams

                if generation_config is not None or use_generation_config:
                    # Use generation config if enabled
                    forced_bos_token_id = (
                        getattr(generation_config, "forced_bos_token_id", None)
                        if generation_config
                        else None
                    )
                    if forced_bos_token_id is None:
                        forced_bos_token_id = getattr(tokenizer, "bos_token_id", None)
                    if forced_bos_token_id is not None:
                        logger.debug(
                            f"Using generation config - forced_bos_token_id: {forced_bos_token_id}"
                        )
                        generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

            inputs = tokenizer(
                TextSummarizer._preprocess_text(
                    request.input, model_config.prepend_text
                ),
                return_tensors="pt",
            )
            temperature = 0.0
            if request.temperature is not None and request.temperature > 0.0:
                temperature = request.temperature
            else:
                temperature = getattr(model_config, "temperature", 0.0)

            if temperature is not None and temperature > 0.0:
                logger.debug(f"Using temperature: {temperature} for summarization")
                generate_kwargs["temperature"] = temperature

            summary_ids = model.generate(**inputs, **generate_kwargs)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            if not summary:
                logger.warning("Empty summary generated.")
            else:
                logger.debug("Seq2SeqLM summarization completed successfully")
            summaryResults.summary = summary.strip()
        except Exception as e:
            logger.exception("Seq2SeqLM summarization failed")
            summaryResults.success = False
            summaryResults.message = f"Error generating summarization: {str(e)}"
        finally:
            logger.debug(f"Seq2SeqLM summarization completed successfully")
            return summaryResults

    @staticmethod
    def _summarizeOther(
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        tokenizer: Any,
        special_tokens: Set[str],
        model_config: OnnxConfig,
        request: SummarizationRequest,
        tokenizer_path: str,
    ) -> SummaryResults:
        """
        Summarize using ONNX encoder/decoder sessions.
        """
        summaryResults: SummaryResults = SummaryResults(
            summary="", message="", success=True
        )

        try:
            logger.debug(
                f"Summarizing text using ONNX encoder/decoder inside _summarizeOther model: {tokenizer_path}"
            )
            pad_token_id = TextSummarizer._get_key_value_from_generation_dict(
                "pad_token_id", model_config, tokenizer_path, True
            )
            eos_token_id = TextSummarizer._get_key_value_from_generation_dict(
                "eos_token_id", model_config, tokenizer_path, True
            )
            max_length = TextSummarizer._get_key_value_from_generation_dict(
                "max_length", model_config, tokenizer_path, True
            )
            decoder_start_token_id = TextSummarizer._get_key_value_from_generation_dict(
                "decoder_start_token_id", model_config, tokenizer_path, True
            )
            gen_config_dict = TextSummarizer._get_generation_config(tokenizer_path)

            if "decoder_start_token_id" not in gen_config_dict or gen_config_dict[
                "decoder_start_token_id"
            ] is (None or TextSummarizer._MISSING_DECODER_START_TOKEN_ID_KEY):
                auto_config = TextSummarizer._get_auto_config(tokenizer_path)

                model_type = (
                    getattr(auto_config, "model_type", None)
                    or tokenizer.__class__.__name__.lower()
                    or "unknown"
                )
                logger.debug(
                    f"Summarizing with model type: {model_type}, model path: {tokenizer_path}"
                )
                logger.debug(
                    "decoder_start_token_id not found in generation config, checking model config"
                )
                if model_type in ["bart", "bert"]:
                    decoder_start_token_id = eos_token_id
                elif "t5" in model_type:
                    decoder_start_token_id = pad_token_id
                else:
                    decoder_start_token_id = pad_token_id

                gen_config_dict["decoder_start_token_id"] = decoder_start_token_id

            decoder_start_token_id = gen_config_dict["decoder_start_token_id"]

            logger.debug(
                f"Using pad_token_id={pad_token_id}, eos_token_id={eos_token_id}, decoder_start_token_id={decoder_start_token_id} for summarization"
            )
            input_names = model_config.inputnames
            output_names = model_config.outputnames
            decoder_input_names = model_config.decoder_inputnames

            logger.debug(
                f"Input names: {input_names}, Output names: {output_names}, Decoder input names: {decoder_input_names}"
            )

            input_text = TextSummarizer._preprocess_text(
                request.input, model_config.prepend_text
            )

            # Tokenize input
            inputs = tokenizer(
                input_text, return_tensors="np", truncation=True, max_length=max_length
            )
            # Prepare ONNX input dict dynamically
            onnx_inputs = {}
            for name, key in input_names.__dict__.items():
                if key and key in inputs:
                    onnx_inputs[key] = inputs[key]
            # Handle token_type_ids if present in config
            if hasattr(input_names, "tokentype") and getattr(
                input_names, "tokentype", None
            ):
                seq_len = inputs[getattr(input_names, "input", "input_ids")].shape[1]
                token_type_ids = np.zeros((1, seq_len), dtype=np.int64)
                onnx_inputs[getattr(input_names, "tokentype")] = token_type_ids

            # Encoder pass
            encoder_outputs = encoder_session.run(None, onnx_inputs)
            TextSummarizer._log_onnx_outputs(encoder_outputs, encoder_session)

            decoder_input_name = getattr(decoder_input_names, "input", None) or getattr(
                input_names, "input", None
            )
            decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)

            decoder_inputs = {}

            decoder_inputs[
                getattr(decoder_input_names, "encoder_output", "encoder_hidden_states")
            ] = encoder_outputs[0]
            decoder_inputs[decoder_input_name] = decoder_input_ids
            if (
                hasattr(input_names, "mask")
                and getattr(input_names, "mask", None)
                and getattr(input_names, "mask", None) in inputs
            ):
                decoder_inputs[
                    getattr(decoder_input_names, "mask", "encoder_attention_mask")
                ] = inputs[getattr(input_names, "mask")]

            decoder_inputs = {k: v for k, v in decoder_inputs.items() if v is not None}
            for k, v in decoder_inputs.items():
                if hasattr(v, "shape"):
                    logger.debug(f"decoder_inputes[{k!r}] shape: {v.shape}")
                else:
                    logger.debug(f"decoder_inputes[{k!r}] value: {v!r}")
            logger.debug(
                f"Decoder inputs: {decoder_inputs.keys()}, values: {[v.shape if hasattr(v, 'shape') else type(v) for v in decoder_inputs.values()]}"
            )
            temperature = 0.0
            if request.temperature is not None and request.temperature > 0.0:
                temperature = request.temperature
            else:
                temperature = getattr(model_config, "temperature", 0.0)

            # Greedy decoding loop (if logits output)
            is_logits = TextSummarizer._is_logits_output(
                encoder_outputs, decoder_session
            ) or getattr(model_config, "logits", False)
            if is_logits:
                summary_ids = []

                logger.debug(
                    f"Starting greedy decoding with max_length={max_length}, eos_token_id={eos_token_id}"
                )
                for _ in range(max_length):
                    # Update decoder inputs with current input ids
                    decoder_inputs[decoder_input_name] = decoder_input_ids
                    try:
                        decoder_outputs = decoder_session.run(None, decoder_inputs)
                    except Exception:
                        logger.exception("Decoder ONNX inference error")
                        break
                    logits_arr = decoder_outputs[0]  # shape: (1, cur_len, vocab_size)
                    logits = logits_arr[:, -1, :][0]  # shape: (1, cur_len, vocab_size)
                    # Take the last token's logits and argmax over vocab
                    if temperature > 0.0:
                        probs = softmax(logits / temperature)
                        next_token_id = int(np.random.choice(len(probs), p=probs))
                    else:
                        # Greedy decoding
                        next_token_id = int(np.argmax(logits))

                    logger.debug(
                        f"Step: {_}, next_token_id: {next_token_id}, eos_token_id: {eos_token_id}"
                    )

                    summary_ids.append(next_token_id)
                    if next_token_id == eos_token_id:
                        logger.debug("EOS token generated, breaking loop.")
                        break
                    decoder_input_ids = np.concatenate(
                        [decoder_input_ids, [[next_token_id]]], axis=1
                    )
                output_ids = np.array(summary_ids)
            else:
                # Single pass, output is token ids
                try:
                    decoder_outputs = decoder_session.run(None, decoder_inputs)
                except Exception:
                    logger.exception("Decoder ONNX inference error")
                    output_ids = np.array([])
                else:
                    output_ids = decoder_outputs[0].astype(np.int64)

            output_ids = np.squeeze(output_ids)
            if np.any(output_ids < 0):
                negatives = output_ids[output_ids < 0]
                logger.error(f"Negative token IDs found: {negatives}")
            else:
                logger.debug(f"About to call tokenizer.decode")
                summary = tokenizer.decode(output_ids, skip_special_tokens=True)
                summary = TextSummarizer._remove_special_tokens(summary, special_tokens)
                if not summary:
                    logger.warning("Empty summary generated.")
                summaryResults.summary = summary.strip()
        except Exception as e:
            logger.exception("ONNX summarization failed")
            summaryResults.success = False
            summaryResults.message = f"Error generating summarization: {str(e)}"
        finally:
            logger.debug(f"ONNX summarization completed successfully")
            return summaryResults


# HINTS:
# - Use `use_generation_config` in your config to control if generation_config is applied at inference.
# - Always initialize `summary = ""` at the start of summarization methods to avoid reference errors.
# - Set `response_output.results` only once in the `finally` block for consistency.
# - For ONNX models, only pass numpy arrays/tensors as ONNX session inputs (not config values like bool/int/str).
# - Check for negative token IDs in ONNX outputs and log a warning if found.
# - Use `forced_bos_token_id` (not `rced_bos_token_id`) for logging and config.
# - Remove special tokens after decoding to clean up the summary.
# - Use try/except/finally to ensure robust error handling and logging.
# - For batch summarization, use `summarize_batch`, which calls `summarize` for each input.
# - When using generation_config, if a key is missing as an attribute, load the raw JSON as a dict for fallback.
# - Use `auto_config` as a fallback for token IDs and model type if not found in configs.
# - Only include tensor inputs in ONNX input dicts; use generation config values for controlling logic, not as ONNX inputs.
# - Clear thread-local tokenizer cache in tests to ensure patching works as expected.
