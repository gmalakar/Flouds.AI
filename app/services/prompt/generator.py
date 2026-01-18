# =============================================================================
# File: generator.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Text generation strategies for prompt service (seq2seq, encoder-only, ONNX)."""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np
import onnxruntime as ort

from app.config.onnx_config import OnnxConfig
from app.logger import get_logger
from app.models.prompt_request import PromptRequest
from app.services.cache_registry import get_generation_cache
from app.services.prompt.models import DEFAULT_MAX_LENGTH, DEFAULT_VOCAB_SIZE, SummaryResults
from app.services.prompt.text_utils import capitalize_sentences, remove_special_tokens

# log sanitization not required in generator module

logger = get_logger(__name__)

# Note: This module contains complex generation logic. For the refactoring phase,
# these methods are preserved from the original prompt_service.py with minimal changes.
# Future optimization: Extract sampling logic to separate sampling.py module.


def run_seq2seq_generation(
    model: Any,
    tokenizer: Any,
    model_config: OnnxConfig,
    request: PromptRequest,
) -> SummaryResults:
    """Generate text using Seq2SeqLM model from Optimum/Transformers.

    Args:
        model: ORTModelForSeq2SeqLM instance
        tokenizer: Tokenizer instance
        model_config: ONNX configuration
        request: Prompt request

    Returns:
        SummaryResults with generated text or error message
    """
    from app.services.prompt.parameters import build_generation_params
    from app.services.prompt.resource_manager import check_memory_and_clear_cache
    from app.services.prompt.text_utils import prepare_input_text

    # Preemptive memory check
    check_memory_and_clear_cache()

    try:
        generate_kwargs: Dict[str, Any] = build_generation_params(model_config, request)

        # Some ONNX models loaded via optimum do not support PKV cache reuse.
        try:
            mod = getattr(model.__class__, "__module__", "") or ""
            if "optimum" in mod or mod.startswith("optimum"):
                generate_kwargs.setdefault("use_cache", False)
        except Exception:
            pass

        input_text = prepare_input_text(request, getattr(model_config, "prepend_text", None))
        inputs = tokenizer(input_text, return_tensors="pt")

        # Use GenerationConfig to avoid deprecation warning
        try:
            from transformers import GenerationConfig

            generation_config = GenerationConfig(**generate_kwargs)
            summary_ids = model.generate(**inputs, generation_config=generation_config)
        except ImportError:
            summary_ids = model.generate(**inputs, **generate_kwargs)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        summary = capitalize_sentences(summary)

        if not summary:
            logger.warning(f"Empty output generated for input: {input_text[:100]}")

        return SummaryResults(summary=summary, message="", success=True)

    except MemoryError:
        logger.error("Out of memory during Seq2Seq generation")
        try:
            from app.utils.cache_manager import CacheManager

            CacheManager.clear_all_caches()
        except Exception:
            pass
        return SummaryResults(
            summary="",
            message="Out of memory: input text too large. Please reduce size and try again.",
            success=False,
        )
    except Exception as e:
        logger.exception("Seq2SeqLM generation failed")
        return SummaryResults(summary="", message=f"Error generating text: {str(e)}", success=False)


def run_encoder_only_generation(
    model_path: str,
    tokenizer: Any,
    model_config: OnnxConfig,
    request: PromptRequest,
) -> SummaryResults:
    """Generate text from encoder-only model.

    Args:
        model_path: Path to model
        tokenizer: Tokenizer instance
        model_config: ONNX configuration
        request: Prompt request

    Returns:
        SummaryResults with generated text or error message
    """
    from app.services.prompt.config import get_token_config
    from app.services.prompt.resource_manager import check_memory_and_clear_cache
    from app.services.prompt.text_utils import prepare_input_text

    check_memory_and_clear_cache()

    try:
        token_config = get_token_config(model_config, tokenizer)
        input_text = prepare_input_text(request, getattr(model_config, "prepend_text", None))

        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        generated_ids = cast(Any, model_path).generate(
            **inputs,
            max_length=token_config.get("max_length", DEFAULT_MAX_LENGTH),
            num_beams=getattr(model_config, "num_beams", 1),
        )

        output_ids = generated_ids[0, inputs["input_ids"].shape[1] :]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return SummaryResults(summary=output_text, message="", success=True)

    except Exception as e:
        logger.exception("Encoder-only generation failed")
        return SummaryResults(
            summary="",
            message=f"Error generating text: {str(e)}",
            success=False,
        )


def run_onnx_generation(
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    tokenizer: Any,
    special_tokens: Set[str],
    model_config: OnnxConfig,
    request: PromptRequest,
    encoder_path: Optional[str] = None,
) -> SummaryResults:
    """Generate text using ONNX encoder/decoder sessions.

    Args:
        encoder_session: ONNX encoder inference session
        decoder_session: ONNX decoder inference session
        tokenizer: Tokenizer instance
        special_tokens: Set of special tokens to remove
        model_config: ONNX configuration
        request: Prompt request
        encoder_path: Optional path to encoder model

    Returns:
        SummaryResults with generated text or error message
    """
    from app.services.prompt.config import get_token_config
    from app.services.prompt.resource_manager import check_memory_and_clear_cache
    from app.services.prompt.text_utils import prepare_input_text

    check_memory_and_clear_cache()

    try:
        token_config = get_token_config(model_config, tokenizer)
        input_text = prepare_input_text(request, getattr(model_config, "prepend_text", None))

        # Tokenize and run encoder
        inputs = tokenizer(
            input_text,
            return_tensors="np",
            truncation=True,
            max_length=token_config["max_length"],
        )
        encoder_outputs = _run_encoder(encoder_session, inputs, model_config, encoder_path)

        # Generate tokens
        summary_ids = _generate_tokens(
            decoder_session,
            encoder_outputs,
            inputs,
            token_config,
            model_config,
            request,
        )

        # Decode and process output
        summary = _decode_output(summary_ids, tokenizer, special_tokens, input_text)
        return SummaryResults(summary=summary, message="", success=True)

    except MemoryError:
        logger.error("Out of memory during ONNX generation")
        try:
            from app.utils.cache_manager import CacheManager

            CacheManager.clear_all_caches()
        except Exception:
            pass
        return SummaryResults(
            summary="",
            message="Out of memory: input text too large. Please reduce size and try again.",
            success=False,
        )
    except Exception as e:
        logger.exception("ONNX generation failed")
        return SummaryResults(summary="", message=f"Error generating text: {str(e)}", success=False)


# ============================================================================
# ONNX-Specific Helper Functions
# ============================================================================


def _run_encoder(
    encoder_session: ort.InferenceSession,
    inputs: Dict,
    model_config: OnnxConfig,
    encoder_path: Optional[str] = None,
) -> Any:
    """Run encoder inference with proper input handling.

    NOTE: This method contains complex logic for handling dynamic ONNX inputs.
    Future refactoring: Extract to encoder_helper.py for clarity.
    """
    input_names = model_config.inputnames
    onnx_inputs = {}

    # Prepare input IDs
    try:
        in_ids = np.ascontiguousarray(np.asarray(inputs["input_ids"])).astype(np.int64)
    except Exception:
        in_ids = inputs["input_ids"]

    onnx_inputs[getattr(input_names, "input", "input_ids")] = in_ids

    # Prepare attention mask if present
    if "attention_mask" in inputs:
        try:
            mask = np.ascontiguousarray(np.asarray(inputs["attention_mask"])).astype(np.int64)
        except Exception:
            mask = inputs["attention_mask"]
        onnx_inputs[getattr(input_names, "mask", "attention_mask")] = mask

    # Handle optional inputs (position_ids, past_key_values, etc.)
    try:
        sess_inputs = encoder_session.get_inputs()
        seq_len = int(np.asarray(inputs["input_ids"]).shape[1]) if "input_ids" in inputs else 1

        for sin in sess_inputs:
            name = sin.name
            if name in onnx_inputs:
                continue

            # Position IDs: arange sequence
            if "position_ids" in name.lower():
                try:
                    pos_arr = np.arange(seq_len, dtype=np.int64)[None, :]
                    onnx_inputs[name] = pos_arr
                    continue
                except Exception:
                    onnx_inputs[name] = np.zeros((1, seq_len), dtype=np.int64)
                    continue

            # Past key values and cache tensors
            if any(s in name.lower() for s in ["past_key_values", "past", ".key", ".value"]):
                try:
                    shape = [
                        d if isinstance(d, int) else 1 for d in (getattr(sin, "shape", []) or [])
                    ]
                    if not shape:
                        shape = [1, seq_len]
                    onnx_inputs[name] = np.zeros(tuple(shape), dtype=np.float32)
                    continue
                except Exception:
                    pass
    except Exception:
        pass

    # Run encoder
    return encoder_session.run(None, onnx_inputs)


def _generate_tokens(
    decoder_session: ort.InferenceSession,
    encoder_outputs: Any,
    inputs: Dict,
    token_config: Dict[str, int],
    model_config: OnnxConfig,
    request: PromptRequest,
    tokenizer: Optional[Any] = None,
) -> List[int]:
    """Generate token sequence using decoder in autoregressive loop.

    NOTE: This method uses caching and advanced sampling. Future refactoring:
    Extract sampling logic to sampling.py module.
    """
    decoder_input_names = model_config.decoder_inputnames
    decoder_input_ids = np.array([[token_config["decoder_start_token_id"]]], dtype=np.int64)
    # Normalize temperature to concrete float before numeric ops
    temperature_raw = request.temperature or getattr(model_config, "temperature", 0.0)
    try:
        temperature: float = float(temperature_raw or 0.0)
    except Exception:
        temperature = 0.0

    summary_ids: List[int] = []

    # Try to use centralized generation cache
    try:
        gen_cache = get_generation_cache()
        cache_key = _build_generation_cache_key(
            decoder_session,
            encoder_outputs,
            inputs,
            token_config,
            model_config,
            request,
            tokenizer,
        )
        if cache_key and gen_cache:
            cached = gen_cache.get(cache_key)
            if cached:
                return list(cached)
    except Exception:
        pass

    # Autoregressive generation loop
    enc0 = np.ascontiguousarray(np.asarray(encoder_outputs[0]))
    step_start = time.time()
    step_timeout = getattr(request, "timeout", 60) / token_config["max_length"]

    for step in range(token_config["max_length"]):
        if time.time() - step_start > step_timeout * (step + 1):
            logger.warning(f"ONNX generation step timeout at step {step}")
            break

        decoder_inputs = {
            getattr(decoder_input_names, "encoder_output", "encoder_hidden_states"): enc0,
            getattr(decoder_input_names, "input", "input_ids"): decoder_input_ids,
        }

        if "attention_mask" in inputs:
            try:
                mask_arr = np.ascontiguousarray(np.asarray(inputs["attention_mask"])).astype(
                    np.int64
                )
                decoder_inputs[getattr(decoder_input_names, "mask", "encoder_attention_mask")] = (
                    mask_arr
                )
            except Exception:
                decoder_inputs[getattr(decoder_input_names, "mask", "encoder_attention_mask")] = (
                    inputs["attention_mask"]
                )

        try:
            decoder_outputs = decoder_session.run(None, decoder_inputs)
        except Exception as e:
            logger.error(f"Decoder inference error: {e}")
            break

        # Sample next token
        dec_out0 = np.ascontiguousarray(np.asarray(decoder_outputs[0]))
        top_k = getattr(model_config, "top_k", None)
        top_p = getattr(model_config, "top_p", None)
        repetition_penalty = getattr(model_config, "repetition_penalty", None)

        next_token_id = _sample_next_token(
            dec_out0,
            temperature,
            top_k,
            top_p,
            summary_ids,
            repetition_penalty,
        )

        # Validate token
        vocab_size = getattr(model_config, "vocab_size", DEFAULT_VOCAB_SIZE)
        if next_token_id < 0 or next_token_id >= vocab_size:
            logger.warning(f"Invalid token ID generated: {next_token_id}")
            break

        summary_ids.append(next_token_id)

        if next_token_id == token_config["eos_token_id"]:
            break

        decoder_input_ids = np.concatenate([decoder_input_ids, [[next_token_id]]], axis=1)

    return summary_ids


def _sample_next_token(
    logits_arr: Any,
    temperature: float,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generated_ids: Optional[list] = None,
    repetition_penalty: Optional[float] = None,
) -> int:
    """Sample next token from logits with advanced decoding."""
    arr = np.asarray(logits_arr)
    if arr.ndim == 3:
        logits = arr[0, -1, :]
    elif arr.ndim == 2:
        logits = arr[-1, :]
    else:
        logits = arr

    # Apply repetition penalty
    if repetition_penalty and generated_ids:
        logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Greedy or temperature-based sampling
    if temperature <= 0.0:
        return int(np.argmax(logits))

    logits = logits / temperature
    logits = _filter_top_k(logits, top_k)
    probs = _softmax(logits)
    probs = _filter_top_p(probs, top_p)

    return _sample_from_probs(probs)


def _decode_output(
    output_ids: List[int],
    tokenizer: Any,
    special_tokens: Set[str],
    input_text: str,
) -> str:
    """Decode token IDs to text with special token removal and capitalization."""
    if not output_ids:
        logger.warning("No valid tokens generated")
        return ""

    try:
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        decoded = remove_special_tokens(decoded, special_tokens)
        decoded = capitalize_sentences(decoded)
        return decoded
    except Exception as e:
        logger.error(f"Error decoding output: {e}")
        return ""


# ============================================================================
# Sampling Utilities
# ============================================================================


def _apply_repetition_penalty(
    logits: np.ndarray, generated_ids: List[int], repetition_penalty: float
) -> np.ndarray:
    """Apply repetition penalty to logits."""
    try:
        if repetition_penalty == 1.0 or not generated_ids:
            return logits
        out = logits.copy()
        for token_id in set(generated_ids):
            if 0 <= token_id < len(out):
                if out[token_id] < 0:
                    out[token_id] *= repetition_penalty
                else:
                    out[token_id] /= repetition_penalty
        return out
    except Exception:
        return logits


def _filter_top_k(logits: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    """Keep only top_k logits, set others to -inf."""
    try:
        if not top_k or top_k <= 0:
            return logits
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        filtered = np.full_like(logits, -np.inf)
        filtered[top_k_indices] = logits[top_k_indices]
        return filtered
    except Exception:
        return logits


def _filter_top_p(probs: np.ndarray, top_p: Optional[float]) -> np.ndarray:
    """Apply nucleus (top-p) filtering."""
    try:
        if not top_p or top_p >= 1.0:
            return probs
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        filtered = np.zeros_like(probs)
        filtered[sorted_indices[:cutoff]] = sorted_probs[:cutoff]
        return filtered / (filtered.sum() + 1e-10) if filtered.sum() > 0 else probs
    except Exception:
        return probs


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    try:
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()
    except Exception:
        return np.ones_like(logits) / len(logits)


def _sample_from_probs(probs: np.ndarray) -> int:
    """Sample from probability distribution."""
    try:
        return int(np.random.choice(len(probs), p=probs))
    except Exception:
        return int(np.argmax(probs))


def _build_generation_cache_key(
    decoder_session: ort.InferenceSession,
    encoder_outputs: Any,
    inputs: Dict,
    token_config: Dict[str, int],
    model_config: OnnxConfig,
    request: PromptRequest,
    tokenizer: Optional[Any] = None,
) -> Optional[str]:
    """Build deterministic cache key for generation outputs."""
    try:
        model_id = getattr(request, "model", "") or ""
        model_version = getattr(model_config, "model_folder_name", "") or ""
        tokenizer_id = getattr(tokenizer, "name_or_path", None) if tokenizer else None

        parts = {
            "model": model_id,
            "model_version": model_version,
            "tokenizer": tokenizer_id or "",
            "max_length": int(token_config.get("max_length", 0)),
            "decoder_start_token_id": int(token_config.get("decoder_start_token_id", 0)),
            "eos_token_id": int(token_config.get("eos_token_id", 0)),
            "tenant": getattr(request, "tenant_code", None) or "",
        }

        m = hashlib.sha256()
        m.update(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode())

        if isinstance(inputs, dict) and "input_ids" in inputs:
            ids = np.ascontiguousarray(inputs["input_ids"]).astype(np.int64)
            m.update(b"||input_ids||")
            m.update(str(ids.shape).encode())

        return m.hexdigest()
    except Exception:
        return None
