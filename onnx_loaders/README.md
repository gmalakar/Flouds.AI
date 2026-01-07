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
python -m onnx_loaders.export_model --model_for s2s --model_name t5-small --optimize --task seq2seq-lm --use_cache

# Fine-tune T5 embeddings and export (alias --finetune supported)
python -m onnx_loaders.export_model --model_for fe --model_name sentence-transformers/sentence-t5-base --finetune --use_t5_encoder --optimize

# Repack external_data models into single-file ONNX during verification
python -m onnx_loaders.export_model --model_for s2s --model_name facebook/bart-large-cnn --task text2text-generation-with-past --use_cache --pack_single_file
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
- `--task`: Export task (e.g., `seq2seq-lm`, `sequence-classification`, `feature-extraction`) — inferred if omitted
- `--use_cache`: Request KV-cache export for seq2seq models (produces `decoder_with_past_model.onnx`)
- `--use_t5_encoder`: Export T5 encoder-only model (for T5 encoder-based embeddings)
- `--model_folder`: Custom output folder name
- `--onnx_path`: Base output path for ONNX models (default: `../onnx`)
- `--opset_version`: ONNX opset (v2 supports `--opset_version`, default 14)
- `--usev1`: Use legacy v1 exporter path (for compatibility)
- `--finetine` / `--finetune`: Fine-tune T5 embeddings before export (alias supported). The fine-tuned folder is used as the export source.
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
                ├── decoder_with_past_model.onnx  # if --use_cache
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
3. Use `--use_cache` for seq2seq models when you need fast autoregressive generation.
4. Keep `transformers`, `optimum`, and `onnxruntime` versions aligned between export and runtime.
5. Validate exported model inputs/outputs with `onnxruntime.InferenceSession` before deploying.

If you need help upgrading runtime dependencies or running a smoke export, open an issue or
run the example commands in `load_scripts.txt`.