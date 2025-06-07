"""
summarizer_service.py

Provides the TextSummarizer class for ONNX and HuggingFace-based text summarization.
Includes thread-safe caching, dynamic config loading, and robust logging.
"""

import json
import os
import time
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, GenerationConfig

from app.config.config_loader import ConfigLoader
from app.config.onnx_config import OnnxConfig
from app.logger import get_logger
from app.models.summarization_response import SummarizationResponse, SummaryResults
from app.modules.concurrent_dict import ConcurrentDict
from app.setup import APP_SETTINGS

logger = get_logger("summarizer_service")


class TextSummarizer:
    """
    Static class for text summarization using ONNX or HuggingFace models.
    """

    _root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "onnx"))
    _encoder_sessions = ConcurrentDict()
    _decoder_sessions = ConcurrentDict()
    _models = ConcurrentDict()
    _tokenizers = ConcurrentDict()
    _special_tokens = ConcurrentDict()

    @staticmethod
    def _get_model_config(model_to_use: str) -> OnnxConfig:
        return ConfigLoader.get_onnx_config()

    @staticmethod
    def _load_special_tokens(special_tokens_path: str) -> set:
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
    def _preprocess_text(text: str, prepend_text: str = None) -> str:
        """
        Preprocesses input text for summarization.
        """
        return f"{prepend_text}{text}" if prepend_text else text

    @classmethod
    def _get_tokenizer(cls, tokenizer_path: str) -> Any:
        """
        Returns a cached tokenizer for the given vocab path.
        """
        return cls._tokenizers.get_or_add(
            tokenizer_path, lambda: AutoTokenizer.from_pretrained(tokenizer_path)
        )

    @classmethod
    def _get_encoder_session(cls, encoder_model_path: str) -> ort.InferenceSession:
        """
        Returns a cached ONNX encoder session.
        """
        provider = APP_SETTINGS.server.model_session_provider or "CPUExecutionProvider"
        logger.debug(
            f"Getting ONNX encoder session for path: {encoder_model_path} with provider: {provider}"
        )
        providers = [provider]
        return cls._encoder_sessions.get_or_add(
            (encoder_model_path, provider),
            lambda: ort.InferenceSession(encoder_model_path, providers=providers),
        )

    @classmethod
    def _get_decoder_session(cls, decoder_model_path: str) -> ort.InferenceSession:
        """
        Returns a cached ONNX decoder session.
        """
        return cls._decoder_sessions.get_or_add(
            decoder_model_path, lambda: ort.InferenceSession(decoder_model_path)
        )

    @classmethod
    def _get_special_tokens(cls, special_tokens_path: str) -> set:
        """
        Returns cached special tokens for the given path.
        """
        return cls._special_tokens.get_or_add(
            special_tokens_path, lambda: cls._load_special_tokens(special_tokens_path)
        )

    @classmethod
    def get_model(cls, model_to_use_path: str) -> ORTModelForSeq2SeqLM:
        """
        Returns a cached ONNX model for the given path.
        """
        return cls._models.get_or_add(
            model_to_use_path,
            lambda: ORTModelForSeq2SeqLM.from_pretrained(
                model_to_use_path, use_cache=False
            ),
        )

    @classmethod
    def summarize(
        cls,
        model_to_use: str,
        text: str,
        use_optimized: bool = False,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        early_stopping: Optional[bool] = None,
    ) -> SummarizationResponse:
        """
        Summarize the input text using the specified model.
        Returns the summary string, or None if summarization fails.
        """
        response = SummarizationResponse(
            success=True,
            message="Summarization generated successfully",
            model=model_to_use,
            results=SummaryResults(summary=""),
        )
        start_time = time.time()

        try:
            model_config = ConfigLoader.get_onnx_config(model_to_use)
            logger.debug(
                f"Summarizing text using model: {model_to_use} and task: {model_config.summarization_task}..."
            )
            model_to_use_path = os.path.join(
                cls._root_path,
                "models",
                model_config.summarization_task or "seq2seq-lm",
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

            tokenizer = cls._get_tokenizer(model_to_use_path)
            special_tokens = cls._get_special_tokens(special_tokens_path)

            # Allow override of config parameters
            if max_length is None:
                max_length = getattr(model_config, "max_length", 128) or 128

            summary = None
            if model_config.use_seq2seqlm:
                logger.debug(f"Using Seq2SeqLM model: {model_to_use_path}")
                model = cls.get_model(model_to_use_path)
                summary = cls._summarizeSeq2SeqLm(
                    model, tokenizer, model_config, text, max_length, model_to_use_path
                )
            else:
                logger.debug(f"Using encoder/decoder model: {model_to_use_path}")
                encoder_session = cls._get_encoder_session(encoder_model_path)
                decoder_session = cls._get_decoder_session(decoder_model_path)
                summary = cls._summarizeOther(
                    encoder_session,
                    decoder_session,
                    tokenizer,
                    special_tokens,
                    model_config,
                    text,
                    max_length,
                )
            response.results = SummaryResults(summary=summary if summary else "")
        except Exception as e:
            response.success = False
            response.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during summarization")
            return response
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"Summarization completed in {elapsed:.2f} seconds.")
            response.time_taken = elapsed
            return response

    @classmethod
    def summarize_batch(
        cls, model_to_use: str, texts: list[str], **kwargs
    ) -> list[Optional[str]]:
        """
        Summarize a batch of texts.
        """
        return [cls.summarize(model_to_use, text, **kwargs) for text in texts]

    @staticmethod
    def _remove_special_tokens(text: str, special_tokens: set) -> str:
        """
        Removes special tokens from the text.
        """
        for token in special_tokens:
            text = text.replace(token, "")
        return text.strip()

    @staticmethod
    def _summarizeSeq2SeqLm(
        model: ORTModelForSeq2SeqLM,
        tokenizer: Any,
        model_config: OnnxConfig,
        text: str,
        max_length: int,
        model_to_use_path: str,
    ) -> Optional[str]:
        """
        Summarize using a HuggingFace Seq2SeqLM model.
        """
        try:

            num_beams = model_config.num_beams
            early_stopping = model_config.early_stopping
            use_generation_config = model_config.use_generation_config
            inputs = tokenizer(
                TextSummarizer._preprocess_text(text, model_config.prepend_text),
                return_tensors="pt",
            )
            generate_kwargs = {
                "max_length": max_length,
                "early_stopping": early_stopping,
            }
            if num_beams is not None and num_beams > 0:
                generate_kwargs["num_beams"] = num_beams

            if use_generation_config:
                generation_config = GenerationConfig.from_pretrained(model_to_use_path)
                model.generation_config = generation_config
                forced_bos_token_id = (
                    getattr(generation_config, "forced_bos_token_id", None)
                    if use_generation_config
                    else None
                )
                if forced_bos_token_id is None:
                    forced_bos_token_id = getattr(tokenizer, "bos_token_id", None)
                if forced_bos_token_id is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

            summary_ids = model.generate(**inputs, **generate_kwargs)
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            if not summary_text.strip():
                logger.warning("Empty summary generated.")
                return None
            return summary_text.strip()
        except Exception as e:
            logger.exception("Seq2SeqLM summarization failed")
            return None

    @staticmethod
    def _summarizeOther(
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        tokenizer: Any,
        special_tokens: set,
        model_config: OnnxConfig,
        text: str,
        max_length: int,
    ) -> Optional[str]:
        """
        Summarize using ONNX encoder/decoder sessions.
        """
        try:
            input_text = TextSummarizer._preprocess_text(
                text, model_config.prepend_text
            )
            padid = model_config.padid
            input_names = model_config.inputnames
            output_names = model_config.outputnames
            decoder_input_names = model_config.decoder_inputnames

            logger.debug(
                f"Input names: {input_names}, Output names: {output_names}, Decoder input names: {decoder_input_names}"
            )

            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length,
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
            # Debug: log encoder output names and shapes
            output_names_list = [o.name for o in encoder_session.get_outputs()]
            for name, arr in zip(output_names_list, encoder_outputs):
                logger.debug(
                    f"Encoder output: {name}, shape: {arr.shape}, dtype: {arr.dtype}"
                )

            decoder_input_name = getattr(decoder_input_names, "input", None) or getattr(
                input_names, "input", None
            )
            decoder_input_ids = np.array(
                [[getattr(decoder_input_names, "input_ids", padid)]], dtype=np.int64
            )
            # Greedy decoding loop (if logits output)
            if getattr(output_names, "logits", False):
                eos_token_id = (
                    getattr(input_names, "eos_token_id", None)
                    or tokenizer.eos_token_id
                    or 1
                )
                summary_ids = []
                for _ in range(max_length):
                    decoder_inputs = {
                        getattr(
                            decoder_input_names,
                            "encoder_output",
                            "encoder_hidden_states",
                        ): encoder_outputs[0],
                        decoder_input_name: decoder_input_ids,
                    }
                    if (
                        hasattr(input_names, "mask")
                        and getattr(input_names, "mask", None)
                        and getattr(input_names, "mask", None) in inputs
                    ):
                        decoder_inputs[
                            getattr(
                                decoder_input_names, "mask", "encoder_attention_mask"
                            )
                        ] = inputs[getattr(input_names, "mask")]
                    decoder_inputs = {
                        k: v for k, v in decoder_inputs.items() if v is not None
                    }
                    try:
                        outputs = decoder_session.run(None, decoder_inputs)
                    except Exception:
                        logger.exception("Decoder ONNX inference error")
                        return None
                    logits = outputs[0]
                    next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
                    summary_ids.append(next_token_id)
                    if next_token_id == eos_token_id:
                        break
                    decoder_input_ids = np.concatenate(
                        [decoder_input_ids, [[next_token_id]]], axis=1
                    )
                output_ids = np.array(summary_ids)
            else:
                # Single pass, output is token ids
                decoder_inputs = {
                    getattr(
                        decoder_input_names,
                        "encoder_output",
                        "encoder_hidden_states",
                    ): encoder_outputs[0],
                    decoder_input_name: decoder_input_ids,
                }
                if (
                    hasattr(input_names, "mask")
                    and getattr(input_names, "mask", None)
                    and getattr(input_names, "mask", None) in inputs
                ):
                    decoder_inputs[
                        getattr(decoder_input_names, "mask", "encoder_attention_mask")
                    ] = inputs[getattr(input_names, "mask")]
                decoder_inputs = {
                    k: v for k, v in decoder_inputs.items() if v is not None
                }
                logger.debug(
                    f"Decoder inputs: {decoder_inputs.keys()}, shapes: {[v.shape for v in decoder_inputs.values()]}"
                )
                try:
                    outputs = decoder_session.run(None, decoder_inputs)
                except Exception:
                    logger.exception("Decoder ONNX inference error")
                    return None
                output_ids = outputs[0].astype(np.int64)

            output_ids = np.squeeze(output_ids)
            if np.any(output_ids < 0):
                logger.error(f"Negative token IDs found: {output_ids}")
                return None
            summary = tokenizer.decode(output_ids, skip_special_tokens=True)
            summary = TextSummarizer._remove_special_tokens(summary, special_tokens)
            if not summary:
                logger.warning("Empty summary generated.")
                return None
            return summary.strip()
        except Exception:
            logger.exception("ONNX summarization failed")
            return None
