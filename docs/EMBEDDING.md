# Embedding Process Flow and Logic

This document describes the embedding pipeline used in this project (paths refer to `Flouds.Py`), the key caches and constants, deterministic behavior guarantees, and how to run the focused embedding tests.

![Embedding flow diagram](./embedding_flow.svg)

## Overview

- Entry point: `app/services/embedder_service.py` — handles tokenization, ONNX input preparation, ONNX session run, pooling, projection, and final normalization.
- Pooling: `app/utils/pooling_strategies.py` — supports `mean`, `max`, `last`, `cls/first`, and `none` strategies.
- Projection matrix caching and other in-process caches live in: `app/services/cache_registry.py`.
- Numeric constants are centralized in: `app/utils/constants.py` (e.g., `NORM_EPS`, `MASK_SUM_EPS`, `MASK_NEG_INF`, `RANDOM_SEED`).

## Step-by-step Flow

1. Tokenization
   - Input text is tokenized by the configured tokenizer (tokenizer code lives in `app/services` or model-specific utilities).
   - Tokenizer outputs commonly include `input_ids` and `attention_mask`. The embedder makes no assumption about tokenizer internals; it expects standard arrays.

2. Prepare ONNX inputs (`_prepare_onnx_inputs` in `embedder_service.py`)
   - The method resolves ONNX session input names using a robust lookup order:
     1. Config override (if provided).
     2. Exact name match against tokenizer keys.
     3. Case-insensitive name match.
     4. Substring match if nothing else matches.
   - It ensures inputs have a batch dimension (1 as needed) and enforces expected dtypes (e.g., `int64` for token IDs where ONNX requires it).

3. Run ONNX session
   - ONNX runtime session execution returns hidden states / last-layer outputs. The code uses session caching (session objects are held in the cache registry) to avoid recreation.

4. Pooling
   - `PoolingStrategies.apply(...)` consumes the hidden outputs and `attention_mask`. Pooling expects consistent shapes:
     - 3D: `(batch, seq_len, hidden_dim)`
     - 2D: `(seq_len, hidden_dim)`
   - The embedder normalizes/reshapes `attention_mask` as needed to match the expected shape.
   - Mean pooling uses `MASK_SUM_EPS` to avoid division-by-zero for empty masks; max pooling fills masked positions with `MASK_NEG_INF`.

5. Projection
   - After pooling, the pipeline projects pooled embeddings into the application embedding space using a deterministic random projection.
   - Projection matrices are cached centrally in `cache_registry` under keys like `proj:{input_dim}x{projected_dim}`.
   - Projection matrices are stored as `np.float32` to save memory and to make the matmul fast with NumPy.
   - Deterministic recreation uses `RANDOM_SEED` (from `app/utils/constants.py`) so tests that clear the cache can recreate the expected matrix.

6. Normalization
   - After projection, vectors are normalized (L2) using a hybrid approach:
     - **1D vectors**: Uses `np.linalg.norm()` (highly optimized BLAS implementation)
     - **2D batches**: Uses element-wise operations `np.sqrt(np.sum(embedding * embedding, axis=1))` which scale better for large batches
   - Uses central `NORM_EPS` constant to stabilize small norms and handle zero vectors.
   - Provides +1-3% speedup for typical batches, +60% for large batches (>100 rows).

7. Quantization (Optional)
   - Final optional step to reduce storage and memory footprint.
   - Applied **after normalization** to preserve vector magnitudes appropriately.
   - Three quantization types supported:
     - **int8**: Symmetric quantization for normalized embeddings, maps [-1, 1] → [-127, 127], 4x storage reduction
     - **uint8**: Min-max normalization, maps [min, max] → [0, 255], 4x storage reduction
     - **binary**: Sign-based binarization (>0 → 1, ≤0 → -1), 32x storage reduction
   - Controlled by model config (`onnx_config.json`) with request-level overrides.
   - See `tests/test_quantization.py` and `tests/test_quantization_integration.py` for detailed behavior.

## Caching & Determinism

- Cache registry (`app/services/cache_registry.py`) lazily initializes several caches used by the embedder and other services (e.g., ONNX sessions, projection matrices).
- Projection cache keys follow the pattern `proj:{input_dim}x{projected_dim}` to ensure determinism across runs.
- Hashing used for cache keys prefers memory-efficient approaches (e.g., `memoryview` over `tobytes()` where possible); fallback to bytes is available for unsupported dtypes.
- Avoid import-time side effects: services no longer call heavy `ensure_initialized()` at module import. Initialization is lazy and test-friendly.

## Important Constants

- `NORM_EPS` — small epsilon (1e-12) used in normalization to avoid divide-by-zero.
- `MASK_SUM_EPS` — epsilon (1e-9) used when summing attention masks for mean pooling.
- `MASK_NEG_INF` — large negative value (-1e9) used to mask positions for max pooling.
- `RANDOM_SEED` — centralized RNG seed (42) used for deterministic projection matrix creation.

All of the above live in `app/utils/constants.py`.

## Configuration

### Quantization Configuration

Quantization can be configured at two levels:

**1. Model-level (in `onnx_config.json`):**
```json
{
  "model-name": {
    "quantize": true,
    "quantize_type": "int8",
    ...
  }
}
```

**Default Values by Model Type:**
- **Embedding models** (all-MiniLM-L6-v2, e5-base-v2, paraphrase-MiniLM-L6-v2, sentence-t5-base, all-mpnet-base-v2): `quantize: true`, `quantize_type: "int8"` for optimal storage reduction with minimal quality loss
- **Large context models** (pleiaspico): `quantize: false` to prioritize quality, but `quantize_type: "int8"` available for request override
- **Generation models** (t5-small, falconsai_text_summarization, bart-large-cnn, distilgpt2): `quantize: false` to maintain generation quality

**2. Request-level (overrides model config):**
```json
{
  "model": "model-name",
  "input": "text to embed",
  "quantize": true,
  "quantize_type": "binary"
}
```

### Quantization Types Comparison

| Type | Storage | Accuracy Loss | Best For |
|------|---------|---------------|----------|
| **int8** | 4x reduction | ~2-5% | General purpose, normalized embeddings |
| **uint8** | 4x reduction | ~2-5% | Non-normalized embeddings |
| **binary** | 32x reduction | ~10-15% | Ultra-fast similarity search |

## Tests

- Focused tests that exercise the embedding flow and caches:
  - `tests/test_projection_matrix_cache.py` — verifies deterministic projection recreation and cache behavior.
  - `tests/test_embedder_inputs.py` — validates ONNX input batching/dtype rules and attention-mask normalization across shapes.
  - `tests/test_normalization_performance.py` — validates hybrid normalization correctness and edge cases.
  - `tests/test_quantization.py` — unit tests for quantization logic (18 tests).
  - `tests/test_quantization_integration.py` — integration tests for quantization pipeline (17 tests).

Run the embedding-focused tests with:

```powershell
python -m pytest tests/test_embedder_inputs.py -q
python -m pytest tests/test_projection_matrix_cache.py -q
python -m pytest tests/test_normalization_performance.py -q
python -m pytest tests/ -k "quantization" -q
```

Or run the whole test suite:

```powershell
python -m pytest -q
```

## Implementation Pointers (file map)

- Embedding orchestration: `app/services/embedder_service.py`
- Cache registry & caches: `app/services/cache_registry.py`
- Pooling strategies: `app/utils/pooling_strategies.py`
- Numeric constants: `app/utils/constants.py`
- Model configuration: `app/config/onnx_config.py`, `app/config/onnx_config.json`
- Request models: `app/models/embedding_request.py`
- Utilities (safe hashing, math ops): `app/utils/*`
- Tests: `tests/test_embedder_inputs.py`, `tests/test_projection_matrix_cache.py`, `tests/test_normalization_performance.py`, `tests/test_quantization.py`, `tests/test_quantization_integration.py`
- Performance benchmarks: `scripts/benchmark_normalization.py`
- Documentation: `docs/NORMALIZATION_OPTIMIZATION.md`

## Best Practices & Notes

- Keep heavy initialization lazy to make unit tests isolated and fast.
- When adding new model ONNXs, add input-name mappings to the config if the automatic lookup fails.
- Prefer `np.float32` for large parameter caches (projection matrices) to save memory.
- Keep RNG seeds deterministic when the projection needs repeatability for tests; production may override if desired.

If you'd like, I can also:
- Add a short diagram illustrating the flow.
- Add example config snippets showing how to override ONNX input names.
- Convert this README into `docs/EMBEDDING.md` and link it from `README.md`.
