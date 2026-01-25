# =============================================================================
# File: test_prompt_service_generation_cache.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from types import SimpleNamespace

import numpy as np

from app.services.cache_registry import clear_generation_cache, get_generation_cache
from app.services.prompt import generator


def test_generate_tokens_returns_cached_result_and_does_not_mutate_cache():
    """Verify that when a generation cache entry exists, `_generate_tokens`
    returns the cached list (as a shallow copy) and does not mutate the stored
    cache value when the caller modifies the returned list.
    """
    # Ensure a clean cache state
    clear_generation_cache()

    # Build minimal inputs expected by the key builder
    inputs = {
        "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1]], dtype=np.int8),
    }

    # Simple encoder output (first output will be hashed)
    encoder_outputs = [np.zeros((1, 3, 8), dtype=np.float32)]

    token_config = {"max_length": 10, "decoder_start_token_id": 0, "eos_token_id": 2}

    # model_config only needs attributes read by the key builder; use SimpleNamespace
    model_config = SimpleNamespace(
        model_folder_name="test-model",
        vocab_size=1000,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        temperature=0.0,
    )
    # Provide decoder input name mappings expected by `_generate_tokens`
    model_config.decoder_inputnames = SimpleNamespace(
        encoder_output="encoder_hidden_states",
        input="input_ids",
        mask="encoder_attention_mask",
    )

    request = SimpleNamespace(model="test-model", temperature=0.0, seed=None, tenant_code=None)

    # Build the cache key using the same helper the service uses
    cache_key = generator._build_generation_cache_key(
        decoder_session=None,
        encoder_outputs=encoder_outputs,
        inputs=inputs,
        token_config=token_config,
        model_config=model_config,
        request=request,
        tokenizer=None,
    )

    assert cache_key is not None, "Cache key should be generated successfully"

    gen_cache = get_generation_cache()
    assert gen_cache is not None

    # Put a known value into the cache
    gen_cache.put(cache_key, [42, 43, 44])

    decoder_session = object()

    # Call the generation routine; because the cache contains the key, it
    # should return the cached list rather than running the decoder.
    result = generator._generate_tokens(
        decoder_session,
        encoder_outputs,
        inputs,
        token_config,
        model_config,
        request,
        tokenizer=None,
    )

    assert result == [42, 43, 44]

    # Mutate the returned list and ensure the stored cache value is unchanged
    result.append(99)
    stored = gen_cache.get(cache_key)
    assert stored == [42, 43, 44], "Cache value must not be mutated by caller"
