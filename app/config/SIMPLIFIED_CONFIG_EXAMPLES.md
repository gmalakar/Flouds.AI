# Simplified ONNX Configuration Examples

This file shows how much simpler your `onnx_config.json` can be with auto-detection enabled.

## Before Auto-Detection (Full Manual Config)

```json
"all-MiniLM-L6-v2": {
    "dimension": 384,
    "max_length": 256,
    "tasks": ["embedding"],
    "normalize": true,
    "pooling_strategy": "mean",
    "chunk_logic": "sentence",
    "chunk_overlap": 1,
    "legacy_tokenizer": false,
    "lowercase": false,
    "remove_emojis": false,
    "force_pooling": true,
    "inputnames": {
        "input": "input_ids",
        "mask": "attention_mask",
        "tokentype": "token_type_ids"
    },
    "outputnames": {
        "output": "last_hidden_state"
    },
    "encoder_onnx_model": "model.onnx",
    "use_optimized": true,
    "encoder_optimized_onnx_model": "model_optimized.onnx"
}
```

## After Auto-Detection (Minimal Config) ✨

```json
"all-MiniLM-L6-v2": {
    "max_length": 256,
    "tasks": ["embedding"],
    "normalize": true,
    "pooling_strategy": "mean",
    "chunk_logic": "sentence",
    "chunk_overlap": 1,
    "force_pooling": true,
    "encoder_onnx_model": "model.onnx",
    "use_optimized": true,
    "encoder_optimized_onnx_model": "model_optimized.onnx"
}
```

**Removed (Auto-Detected):**
- ❌ `dimension` → Auto-detected as 384
- ❌ `inputnames` → Auto-detected as `input_ids`, `attention_mask`
- ❌ `outputnames` → Auto-detected from model outputs
- ❌ `legacy_tokenizer`, `lowercase`, `remove_emojis` → Default to false

**Lines Saved:** 11 lines (35% reduction)

---

## Language Model Example

### Before
```json
"distilgpt2": {
    "dimension": 768,
    "max_length": 512,
    "min_length": 10,
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "bos_token_id": 50256,
    "vocab_size": 50257,
    "tasks": ["language_model"],
    "chunk_logic": "sentence",
    "chunk_overlap": 1,
    "legacy_tokenizer": false,
    "lowercase": false,
    "remove_emojis": false,
    "force_pooling": false,
    "inputnames": {
        "input": "input_ids",
        "mask": "attention_mask"
    },
    "outputnames": {
        "output": "logits"
    },
    "encoder_onnx_model": "model.onnx",
    "use_optimized": true,
    "encoder_optimized_onnx_model": "model_optimized.onnx",
    "special_tokens_map_path": "special_tokens_map.json",
    "num_beams": 1,
    "do_sample": true,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "use_seq2seqlm": false,
    "encoder_only": true
}
```

### After ✨
```json
"distilgpt2": {
    "max_length": 512,
    "min_length": 10,
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "bos_token_id": 50256,
    "tasks": ["language_model"],
    "chunk_logic": "sentence",
    "encoder_onnx_model": "model.onnx",
    "use_optimized": true,
    "encoder_optimized_onnx_model": "model_optimized.onnx",
    "special_tokens_map_path": "special_tokens_map.json",
    "num_beams": 1,
    "do_sample": true,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "encoder_only": true
}
```

**Removed (Auto-Detected):**
- ❌ `dimension` → Auto-detected as 768
- ❌ `vocab_size` → Auto-detected as 50257
- ❌ `inputnames` → Auto-detected from model
- ❌ `outputnames` → Auto-detected from model
- ❌ `legacy_tokenizer`, `lowercase`, `remove_emojis`, `force_pooling`, `chunk_overlap`, `use_seq2seqlm` → Defaults

**Lines Saved:** 13 lines (40% reduction)

---

## New Model Template (Minimal Setup)

### For Embedding Models
```json
"your-new-embedding-model": {
    "max_length": 512,
    "tasks": ["embedding"],
    "normalize": true,
    "pooling_strategy": "mean",
    "chunk_logic": "sentence",
    "encoder_onnx_model": "model.onnx"
}
```
**That's it!** Only 6 properties needed. Everything else auto-detected.

### For Language Models
```json
"your-new-gpt-model": {
    "max_length": 512,
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "tasks": ["language_model"],
    "chunk_logic": "sentence",
    "encoder_onnx_model": "model.onnx",
    "encoder_only": true
}
```
**Only 7 properties!** Much simpler than before.

---

## Migration Guide

### Step 1: Identify Auto-Detectable Properties
Review your config and mark these for removal:
- ✅ `dimension` (unless you want to override)
- ✅ `inputnames` (unless special case like T5)
- ✅ `outputnames` (unless specific output selection)
- ✅ `vocab_size` (for LLMs)

### Step 2: Remove Default Values
Remove properties that match defaults:
- ✅ `legacy_tokenizer: false`
- ✅ `lowercase: false`
- ✅ `remove_emojis: false`
- ✅ `force_pooling: false`
- ✅ `chunk_overlap: 0` or `1`
- ✅ `use_seq2seqlm: false`

### Step 3: Keep Required Behavioral Settings
**Always keep these:**
- ✅ `max_length` - Defines chunking limit
- ✅ `tasks` - Model purpose (list of capabilities such as `embedding`, `summarization`, `language_model`, `prompt`)
- ✅ `normalize` - Embedding normalization
- ✅ `pooling_strategy` - How to pool embeddings
- ✅ `chunk_logic` - Text splitting strategy
- ✅ Token IDs for generation models
- ✅ Model file paths

### Step 4: Test Your Minimal Config
```bash
python -m pytest tests/test_embedder_service.py -v
python -m pytest tests/test_prompt_service.py -v
```

---

## Benefits of Minimal Config

✅ **Less Maintenance:** Fewer lines to update when changing models  
✅ **Fewer Errors:** Auto-detection ensures accuracy  
✅ **Easier Onboarding:** New models work with minimal setup  
✅ **Self-Documenting:** Config shows only behavioral choices  
✅ **Future-Proof:** Model upgrades don't require config changes  

---

## When to Override Auto-Detection

### Override `dimension` when:
- You want smaller embeddings than native (downsampling)
- You need consistent dimensions across different models

### Override `inputnames` when:
- Using T5 models with decoder inputs
- Model has non-standard input names
- Special tokenizer requirements

### Override `outputnames` when:
- Model has multiple outputs and you need specific one
- Non-standard output tensor names

### Override `vocab_size` when:
- Token validation needs custom threshold
- Model has special vocabulary requirements

---

**Remember:** Auto-detection happens at runtime. Check logs to see what values were detected:
```
INFO: Auto-detected native dimension from ONNX model: 384
INFO: Auto-detected primary output name: last_hidden_state
```
