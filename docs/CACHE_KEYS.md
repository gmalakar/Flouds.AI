# Cache Key Utility

This document describes the `get_model_cache_key` utility and recommended developer practices.

Location
--------

- `app/utils/cache_keys.py`

Purpose
-------

`get_model_cache_key(model_to_use: str) -> str` centralizes canonicalization of model identifiers used as keys in runtime caches. It:

- Resolves models configured with `model_folder_name` to an absolute, validated filesystem path when possible.
- Returns normalized absolute paths for existing filesystem paths passed as identifiers.
- Otherwise returns the provided model identifier string (trimmed).

Rationale
---------

Centralizing this logic makes it safe to change canonicalization rules (for example, to prefer symlink resolution, path hashing, or repo-id normalization) without modifying multiple services. It also avoids import cycles (the utility does not import `BaseNLPService`).

Developer Guidance
------------------

- Import and call the utility directly:

```py
from app.utils.cache_keys import get_model_cache_key

key = get_model_cache_key("t5-small")
```

- If you previously patched `BaseNLPService._get_model_cache_key` in tests, update test patches to mock `app.utils.cache_keys.get_model_cache_key` instead.

Suggested Unit Tests
--------------------

Add tests under `tests/` to cover the following cases:

1. Identifier that resolves via `onnx_config` `model_folder_name` to a path within `APP_SETTINGS.onnx.onnx_path`.
2. Identifier that is an existing absolute path on disk — returns normalized absolute path.
3. Identifier that looks like a path but does not exist — returns the original string trimmed.
4. Malformed or missing config — ensure the function falls back to returning the identifier string and does not raise.

Example test matrix (pytest):

- `test_get_model_cache_key_resolves_config()`
- `test_get_model_cache_key_existing_path(tmp_path)`
- `test_get_model_cache_key_nonexistent_path()`
- `test_get_model_cache_key_invalid_config()`

Notes
-----

The function intentionally swallows exceptions and returns the identifier string as a safe fallback. If you need stricter behavior (raise on invalid config), wrap the utility call and perform additional validation.
