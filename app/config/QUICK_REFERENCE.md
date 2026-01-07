# ONNX Config Quick Reference

## 🔄 Auto-Detected (Can Be Omitted)

| Property | Detected From | Override When |
|----------|---------------|---------------|
| `dimension` | Output shape `[batch, seq, **dim**]` | Want smaller dimension |
| `inputnames` | `session.get_inputs()` | T5 decoder, special inputs |
| `outputnames` | `session.get_outputs()` | Multiple outputs, specific selection |
| `vocab_size` | Logits shape `[batch, seq, **vocab**]` | Custom token validation |

## ✅ Required (Must Specify)

### All Models
- `max_length` - Token limit for chunking
- `encoder_onnx_model` - Model file path
- `chunk_logic` - "sentence" / "paragraph" / "fixed"

### Embedding Models (models with `"embedding"` in `tasks`)
- `tasks`: include `"embedding"`
- `normalize` - true/false
- `pooling_strategy` - "mean"/"max"/"first"/"last"

### Generation Models (models with `"summarization"` or `"language_model"` in `tasks`)
- `tasks`: include `"summarization"` or `"language_model"`
- `pad_token_id`
- `eos_token_id`
- `decoder_start_token_id` (seq2seq only)

## 📝 Minimal Templates

### Embedding Model (6 lines)
```json
{
    "max_length": 512,
    "tasks": ["embedding"],
    "normalize": true,
    "pooling_strategy": "mean",
    "chunk_logic": "sentence",
    "encoder_onnx_model": "model.onnx"
}
```

### Language Model (7 lines)
```json
{
    "max_length": 512,
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "tasks": ["language_model"],
    "chunk_logic": "sentence",
    "encoder_onnx_model": "model.onnx",
    "encoder_only": true
}
```

## 📚 Full Documentation

- **Complete Guide:** `ONNX_CONFIG_GUIDE.md`
- **Examples:** `SIMPLIFIED_CONFIG_EXAMPLES.md`
- **Main Config:** `onnx_config.json`
