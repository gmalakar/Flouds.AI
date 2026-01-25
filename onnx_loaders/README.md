# ONNX Model Export Tools

This directory contains tools for exporting HuggingFace models to ONNX format for use with Flouds AI.

## Files

- `export_model.py` - Main export script with command-line interface
- `export_model_to_onnx.py` - Core export logic and functions
- `batch_export.py` - Batch export script for multiple models
- `load_scripts.txt` - Example export commands
- `README.md` - This documentation
# ONNX Model Export Tools

This directory contains tools for exporting HuggingFace models to ONNX format for use with Flouds AI.

## Overview

- `export_model.py` - Main command-line entrypoint (now uses the consolidated exporter)
- `export_model_consolidated.py` - Unified export logic (preferred)
- `onnx_utils.py` - ORT session helpers, logger, and environment helpers
- `fine_tune_t5.py` - Example fine-tuning helper for T5 embeddings
- `load_scripts.txt` - Example export commands

Legacy modules `export_model_to_onnx.py` and `export_model_to_onnx_v2.py` have been deprecated
and replaced by the consolidated exporter. The CLI `export_model.py` maps flags to the unified
export function and preserves a compatibility switch for legacy behavior.

## Quick Start

### Export Single Model (default: v2 behavior)

```powershell
# Embedding model
python -m onnx_loaders.export_model --model_for fe --model_name sentence-transformers/all-MiniLM-L6-v2 --optimize

# Summarization model (seq2seq)
python -m onnx_loaders.export_model --model_for s2s --model_name t5-small --optimize --task seq2seq-lm

# Fine-tune T5 embeddings and export (alias --finetune supported)
# Note: T5 encoder-only export now requires an explicit `--task feature-extraction` parameter
python -m onnx_loaders.export_model --model_for fe --model_name sentence-transformers/sentence-t5-base --finetune --task feature-extraction --optimize

# Repack external_data models into single-file ONNX during verification
python -m onnx_loaders.export_model --model_for s2s --model_name facebook/bart-large-cnn --task text2text-generation-with-past --pack_single_file
```

### Use legacy v1 behavior

If you need the legacy ORTModel-based export path, pass `--usev1`:

```powershell
python -m onnx_loaders.export_model --model_for fe --model_name t5-small --usev1
```

## CLI Parameters

- `--model_for`: Model type (`fe`=embedding/feature-extraction, `s2s`=seq2seq, `sc`=classification, `llm`=causal LM)
- `--model_name`: HuggingFace model name or local path (e.g., `t5-small`)
- `--optimize`: Enable ONNX optimization (recommended for production)
- `--optimization_level`: Optimization level (integer, default 1)
- `--task`: Export task (e.g., `seq2seq-lm`, `sequence-classification`, `feature-extraction`) — required
KV-cache exports are auto-detected; to request KV-cache behavior for seq2seq models, provide `--task text2text-generation-with-past` (there is no `--use_cache` CLI flag).
- `--model_folder`: Custom output folder name
- `--onnx_path`: Base output path for ONNX models (default: `../onnx`)
- `--opset_version`: ONNX opset (v2 supports `--opset_version`, default 14)
- `--usev1`: Use legacy v1 exporter path (for compatibility)
-- `--finetune`: Fine-tune T5 embeddings before export. The fine-tuned folder is used as the export source.
- `--pack_single_file`: If exported ONNX uses external_data, repack into a single-file model in-place during verification.

## Output Structure

Models are exported to `../onnx/models/{task}/{model_name}/` (relative to the `onnx_loaders` module directory):

```
onnx/models/
├── fe/
│   └── sentence-t5-base/
│       ├── model.onnx
│       ├── tokenizer.json
│       └── config.json
└── s2s/
        └── t5-small/
                ├── encoder_model.onnx
                ├── decoder_model.onnx
                ├── decoder_with_past_model.onnx  # if KV-cache export enabled (auto-detected)
                ├── tokenizer.json
                └── special_tokens_map.json
```

## Troubleshooting

- If ONNX export fails due to protobuf serialization errors, the consolidated exporter will retry
    with a lower `opset` (11) and then attempt the ORTModel fallback. If you see IR/opset mismatch
    errors at inference time, upgrade `onnxruntime` to a version supporting your model's IR (e.g.,
    `onnxruntime>=1.18.0` for IR v11).
- For large models, try exporting without `--optimize` first, then optimize on a machine with more RAM.
- If exported models reference external tensor data (`external_data`), use `--pack_single_file` to convert them
    into single-file ONNX artifacts during verification; ensure associated tensor files are retained if you do not repack.

## Best Practices

1. Use the consolidated CLI (`export_model.py`) which defaults to the v2 behavior and provides
     robust fallbacks.
2. Use `--optimize` for production models, but ensure the host has sufficient memory.
3. KV-cache is auto-detected; supply `--task text2text-generation-with-past` to request KV-cache export for seq2seq models when you need fast autoregressive generation.
4. Keep `transformers`, `optimum`, and `onnxruntime` versions aligned between export and runtime.
5. Validate exported model inputs/outputs with `onnxruntime.InferenceSession` before deploying.

If you need help upgrading runtime dependencies or running a smoke export, open an issue or
run the example commands in `load_scripts.txt`.

## Recent Changes (summary)

- Canonicalized CLI flags to hyphenated forms (e.g. `--model-name`). Legacy underscore-style aliases were removed.
- New CLI flags:
    - `--no-post-process`: Skip optimum post-processing (deduplication) to avoid MemoryError on very large models.
    - `--no-local-prep`: Skip creating a prepared local copy (temp_local) to reduce disk usage during export.
    - `--no-use-external-data-format`: Prefer single-file ONNX where possible (disable external_data shards).
    - `--prune-canonical`: Optionally remove canonical ONNX files when merged artifacts exist.

- `export_v2.py` (v2 exporter wrapper) hardening and behavior:
    - Runs `optimum.exporters.onnx.main_export` inside a sanitized subprocess to isolate native crashes and reduce memory/threading pressure.
    - Writes a short runner script which normalizes legacy `torch_dtype` → modern `dtype` keys before invoking `main_export`.
    - Pre-export best-effort cleanup of stale `onnx_*` temp dirs to free disk space on hosts under pressure.
    - Uses a short temporary working output on Windows to avoid very long external-data paths; artifacts are moved back on success.
    - Retry/fallback sequence on failures:
        1. Retry with `trust_remote_code=True` if remote code execution is required.
        2. If protobuf serialization/encode errors appear, try a fallback opset (intelligently select opset 14 when `scaled_dot_product_attention` is implicated, otherwise fallback to 11).
        3. Detect optimum post-processing MemoryError traces and retry with `no_post_process=True` combined with conservative memory reducers (`dtype=float16`, `use_external_data_format=False`).
        4. Detect OOM/MemoryError and retry with `dtype=float16`, then `float16 + no external_data`, and finally `opset=11 + float16 + no external_data` as a last resort.
    - Clone-based fallback: if allowed, the wrapper will copy/clone the model repo locally (converting HF `owner/name` ids to `https://huggingface.co/<owner>/<name>` clone URLs) and retry with `trust_remote_code=True`.
    - Best-effort cleanup of transient temp dirs on unexpected exceptions (removes `onnx_out_*/`, `onnx_export_*`, `onnx_opt_clone_*`, `onnx_working_*` prefixes when older than a threshold).

- Verification & optimization:
    - Validator invocation is preserved; quick verification runs and may skip model variants that are not present (e.g., `model_with_past.onnx` when not produced).
    - `--pack-single-file` behavior improved to repack external-data into a single-file ONNX during verification when requested.
    - Optimizer/ORTModel load fallbacks added: clone-based optimization loading and conservative handling of optimizer failures.

- CLI safety: `export_model.py` now validates the set of keyword arguments it forwards to the consolidated exporter and will reject unknown parameter names with a clear error (helps catch typos and removed aliases).

## Notes on disk usage

- Large models will create multi-GB external-data shards under `onnx_loaders/onnx/models/...` (these are not system temp files). If you observe sudden disk growth, check that folder first and consider:
    - Using `--no-local-prep` to avoid creating a local prepared snapshot.
    - Using `--no-post-process` to avoid memory-heavy deduplication when trying to export very large LLMs.
    - Using `--no-use-external-data-format` or `--pack-single-file` to keep a single `.onnx` file (may increase peak memory when repacking).
    - Moving `ONNX_PATH` to a larger drive for temporary/export storage.

If you'd like, I can add an automated pruning command or a CI check to remove or archive old exports.