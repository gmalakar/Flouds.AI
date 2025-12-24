# ONNX Model Export Tools

This directory contains tools for exporting HuggingFace models to ONNX format for use with Flouds AI.

## Files

- `export_model.py` - Main export script with command-line interface
- `export_model_to_onnx.py` - Core export logic and functions
- `batch_export.py` - Batch export script for multiple models
- `load_scripts.txt` - Example export commands
- `README.md` - This documentation

## Quick Start

### Export Single Model

```bash
# Embedding model
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/all-MiniLM-L6-v2" --optimize

# Summarization model
python onnx_loaders/export_model.py --model_for "s2s" --model_name "t5-small" --optimize --task "seq2seq-lm" --use_cache
```

### Batch Export All Models

```bash
python onnx_loaders/batch_export.py
```

## Parameters

- `--model_for`: Model type (`fe`=embedding, `s2s`=summarization, `sc`=classification)
- `--model_name`: HuggingFace model name (e.g., `t5-small`)
- `--optimize`: Enable ONNX optimization (recommended)
- `--task`: Export task (`seq2seq-lm`, `feature-extraction`, etc.)
- `--use_cache`: Use KV cache for seq2seq models (faster inference)
- `--model_folder`: Custom output folder name

## Output Structure

Models are exported to `../onnx/models/{task}/{model_name}/`:

```
onnx/models/
├── fe/                          # Feature extraction (embeddings)
│   ├── all-MiniLM-L6-v2/
│   │   ├── model.onnx
│   │   ├── tokenizer.json
│   │   └── config.json
│   └── sentence-t5-base/
└── s2s/                         # Sequence-to-sequence (summarization)
    ├── t5-small/
    │   ├── encoder_model.onnx
    │   ├── decoder_model.onnx
    │   ├── decoder_with_past_model.onnx
    │   ├── tokenizer.json
    │   └── special_tokens_map.json
    └── falconsai_text_summarization/
```

## Troubleshooting

### Tokenizer Compatibility Issues

If you get tokenizer errors like "PyPreTokenizerTypeWrapper", re-export with current transformers version:

```bash
python onnx_loaders/export_model.py --model_for "s2s" --model_name "t5-small" --optimize --task "seq2seq-lm" --use_cache
```

### Memory Issues

For large models, export without optimization first:

```bash
python onnx_loaders/export_model.py --model_for "s2s" --model_name "facebook/bart-large-cnn" --task "seq2seq-lm"
```

### Missing decoder_with_past_model.onnx

Add `--use_cache` flag for seq2seq models:

```bash
python onnx_loaders/export_model.py --model_for "s2s" --model_name "t5-small" --task "seq2seq-lm" --use_cache
```

## Best Practices

1. **Always use `--optimize`** for production models
2. **Use `--use_cache`** for seq2seq models (faster inference)
3. **Test exported models** before deploying
4. **Keep transformers version consistent** between export and runtime
5. **Use batch export** for multiple models to save time