# =============================================================================
# File: test_prompt_service_encoder_output_cache.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import hashlib
import json
from types import SimpleNamespace

import numpy as np

import app.services.cache_registry as cache_registry
from app.services.cache_registry import clear_encoder_output_cache, get_encoder_output_cache
from app.services.prompt import generator
from app.utils.simple_cache import SimpleCache


def test_encoder_output_cache_put_get_promote_float32():
    """Store a float16 encoder output and verify retrieval and promotion."""
    clear_encoder_output_cache()

    enc_cache = get_encoder_output_cache()
    if enc_cache is None:
        # Some test environments may not have the encoder cache initialized
        # due to lazy init; create a SimpleCache so tests can proceed.
        cache_registry.ENCODER_OUTPUT_CACHE = SimpleCache(max_size=128)
        enc_cache = get_encoder_output_cache()
    assert enc_cache is not None

    key = "test-enc-key-1"
    enc0 = np.ones((1, 3, 8), dtype=np.float32) * 1.5
    enc_quant = enc0.astype(np.float16)

    # Put quantized array and get it back
    enc_cache.put(key, enc_quant)
    cached = enc_cache.get(key)
    assert cached is not None
    assert cached.dtype == np.float16

    promoted = np.ascontiguousarray(cached).astype(np.float32)
    assert promoted.dtype == np.float32
    assert promoted.shape == enc0.shape
    assert np.allclose(promoted, enc0, atol=1e-3)


def test_encoder_output_cache_key_build_and_clear():
    """Build the encoder-output cache key the same way the service does,
    store a value, retrieve and clear the cache to verify removal.
    """
    clear_encoder_output_cache()
    enc_cache = get_encoder_output_cache()
    if enc_cache is None:
        cache_registry.ENCODER_OUTPUT_CACHE = SimpleCache(max_size=128)
        enc_cache = get_encoder_output_cache()

    token_config = {"max_length": 10, "decoder_start_token_id": 0, "eos_token_id": 2}

    model_config = SimpleNamespace(model_folder_name="test-model")
    request = SimpleNamespace(model="test-model", tenant_code=None)

    parts = {
        "model": getattr(request, "model", "") or "",
        "model_version": getattr(model_config, "model_folder_name", "") or "",
        "tokenizer": "",
        "max_length": int(token_config.get("max_length", 0)),
        "decoder_start_token_id": int(token_config.get("decoder_start_token_id", 0)),
        "eos_token_id": int(token_config.get("eos_token_id", 0)),
        "tenant": getattr(request, "tenant_code", None) or "",
    }

    m = hashlib.sha256()
    m.update(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode())

    # Simulate tokenizer outputs
    ids = np.ascontiguousarray(np.array([[1, 2, 3]], dtype=np.int64))
    m.update(b"||input_ids||")
    m.update(str(ids.shape).encode())
    m.update(generator._hash_array(ids, quantize_dtype=np.int64).encode())

    mask = np.ascontiguousarray(np.array([[1, 1, 1]], dtype=np.int8))
    m.update(b"||attention_mask||")
    m.update(str(mask.shape).encode())
    m.update(generator._hash_array(mask, quantize_dtype=np.int8).encode())

    enc_key = m.hexdigest()

    enc0 = np.arange(24, dtype=np.float32).reshape((1, 3, 8))
    enc_cache.put(enc_key, enc0.astype(np.float16))

    got = enc_cache.get(enc_key)
    assert got is not None
    promoted = np.ascontiguousarray(got).astype(np.float32)
    assert promoted.shape == enc0.shape

    # Clear and ensure value is removed
    clear_encoder_output_cache()
    got2 = enc_cache.get(enc_key)
    assert got2 is None
