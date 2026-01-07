from types import SimpleNamespace

import numpy as np

from app.services.cache_registry import clear_decode_cache, get_decode_cache
from app.services.prompt_service import PromptProcessor


class DummyTokenizer:
    def __init__(self):
        self.calls = 0

    def decode(self, ids, skip_special_tokens=True):
        self.calls += 1
        # simple deterministic decode: join ids with '-' as string
        return "-".join(str(int(x)) for x in ids)


def test_decode_cache_prevents_repeated_decode_calls():
    clear_decode_cache()
    dec_cache = get_decode_cache()
    # Ensure cache exists
    assert dec_cache is not None

    tokenizer = DummyTokenizer()
    special_tokens = set()
    output_ids = [1, 2, 3]

    # First call should invoke tokenizer.decode
    out1 = PromptProcessor._decode_output(
        output_ids, tokenizer, special_tokens, "input"
    )
    assert out1 == "1-2-3"
    assert tokenizer.calls == 1

    # Second call with same args should be served from cache (no extra decode)
    out2 = PromptProcessor._decode_output(
        output_ids, tokenizer, special_tokens, "input"
    )
    assert out2 == "1-2-3"
    assert tokenizer.calls == 1


def test_decode_cache_separates_different_special_tokens():
    clear_decode_cache()
    tokenizer = DummyTokenizer()
    # Provide special tokens that would be removed by _remove_special_tokens
    special_tokens_a = {"<s>"}
    special_tokens_b = {"<t>"}
    output_ids = [4, 5]

    out_a = PromptProcessor._decode_output(
        output_ids, tokenizer, special_tokens_a, "input"
    )
    assert out_a == "4-5"
    # Different special tokens should produce a separate cache entry
    out_b = PromptProcessor._decode_output(
        output_ids, tokenizer, special_tokens_b, "input"
    )
    assert out_b == "4-5"
    # Two decode calls expected because cache keys differ
    assert tokenizer.calls == 2
