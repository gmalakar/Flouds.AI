# ONNX Model Configuration Guide

This document explains which properties in `onnx_config.json` are required vs optional (auto-detectable).

## 🔄 Auto-Detectable Properties (Optional)

These properties can be automatically detected from the ONNX model files. If omitted, they will be auto-detected at runtime.

### 1. **dimension** 
- **Auto-detected from:** ONNX output shape `[batch, seq_len, dimension]`
- **When to specify:** Override if you want a different dimension than native
- **Example:** `"dimension": 384`
- **Note:** If config dimension > native dimension, native will be used to prevent upsampling

### 2. **inputnames**
- **Auto-detected from:** `session.get_inputs()` - extracts input tensor names
- **When to specify:** Override for special cases (e.g., T5 decoder inputs)
- **Example:**
  ```json
  "inputnames": {
      "input": "input_ids",
      "mask": "attention_mask",
      "position": "position_ids"
  }
  ```
- **Auto-detected mappings:** `input_ids`, `attention_mask`, `token_type_ids`, `position_ids`

### 3. **outputnames**
- **Auto-detected from:** `session.get_outputs()` - extracts output tensor names
- **When to specify:** Override if you need specific output selection
- **Example:**
  ```json
  "outputnames": {
      "output": "last_hidden_state"
  }
  ```

### 4. **vocab_size** (Language Models)
- **Auto-detected from:** Logits output shape `[batch, seq_len, vocab_size]`
- **When to specify:** Override if needed for token validation
- **Example:** `"vocab_size": 50257`
- **Applies to:** Language models (GPT, T5, BART, etc.)

---

## ✅ Required Properties

These properties define model behavior and cannot be auto-detected. They must be specified in the config.

### Core Settings

#### **max_length** (REQUIRED)
- Maximum token length for tokenization and chunking
- **Example:** `"max_length": 256`
- **Common values:** 128-512 for embeddings, 512-2048 for summarization

#### **embedder_task** or **summarization_task** (REQUIRED)
- Defines the model's primary task
- **Values:**
  - `"fe"` - Feature Extraction (embeddings)
  - `"s2s"` - Sequence-to-Sequence (summarization)
  - `"llm"` - Language Model (text generation)
- **Example:** `"embedder_task": "fe"`

### Embedding Model Settings (embedder_task: "fe")

#### **normalize** (REQUIRED)
- Whether to L2 normalize embedding vectors
- **Example:** `"normalize": true`
- **Recommendation:** `true` for cosine similarity

#### **pooling_strategy** (REQUIRED)
- How to pool token embeddings into sentence embedding
- **Values:** `"mean"`, `"max"`, `"first"`, `"last"`
- **Example:** `"pooling_strategy": "mean"`

#### **chunk_logic** (REQUIRED)
- How to split long texts into chunks
- **Values:**
  - `"sentence"` - Split by sentences (NLTK)
  - `"paragraph"` - Split by paragraphs
  - `"fixed"` - Fixed token size chunks
- **Example:** `"chunk_logic": "sentence"`

#### **chunk_overlap** (RECOMMENDED)
- Number of overlapping chunks between segments
- **Example:** `"chunk_overlap": 1`
- **Default:** `0`

#### **force_pooling** (RECOMMENDED)
- Force pooling even if model outputs sentence embeddings
- **Example:** `"force_pooling": true`
- **Default:** `false`

### Generation Model Settings (summarization_task: "s2s" or "llm")

#### **pad_token_id** (REQUIRED)
- Token ID for padding sequences
- **Example:** `"pad_token_id": 0`

#### **eos_token_id** (REQUIRED)
- Token ID for end-of-sequence
- **Example:** `"eos_token_id": 1`

#### **decoder_start_token_id** (REQUIRED for seq2seq)
- Token ID to start decoder generation
- **Example:** `"decoder_start_token_id": 0`

#### **num_beams** (RECOMMENDED)
- Number of beams for beam search
- **Example:** `"num_beams": 4`
- **Default:** `1` (greedy)

#### **early_stopping** (RECOMMENDED)
- Stop generation when all beams hit EOS
- **Example:** `"early_stopping": true`
- **Default:** `false`

### Model File Paths (REQUIRED)

#### **encoder_onnx_model** (REQUIRED)
- Path to encoder/main ONNX model file
- **Example:** `"encoder_onnx_model": "model.onnx"`

#### **decoder_onnx_model** (seq2seq only)
- Path to decoder ONNX model file
- **Example:** `"decoder_onnx_model": "decoder_model.onnx"`

#### **use_optimized** (OPTIONAL)
- Whether to use optimized model versions
- **Example:** `"use_optimized": true`
- **Default:** `false`

#### **encoder_optimized_onnx_model** (if use_optimized: true)
- Path to optimized encoder model
- **Example:** `"encoder_optimized_onnx_model": "model_optimized.onnx"`

### Optional Preprocessing Settings

#### **legacy_tokenizer** (OPTIONAL)
- Use legacy tokenizer implementation
- **Example:** `"legacy_tokenizer": false`
- **Default:** `false`

#### **lowercase** (OPTIONAL)
- Convert text to lowercase before embedding
- **Example:** `"lowercase": false`
- **Default:** `false`

#### **remove_emojis** (OPTIONAL)
- Remove emojis from input text
- **Example:** `"remove_emojis": false`
- **Default:** `false`

---

## 📋 Minimal Configuration Examples

### Minimal Embedding Model (Auto-Detection Enabled)
```json
"my-embedding-model": {
    "max_length": 256,
    "embedder_task": "fe",
    "normalize": true,
    "pooling_strategy": "mean",
    "chunk_logic": "sentence",
    "encoder_onnx_model": "model.onnx"
}
```
**Auto-detected:** `dimension`, `inputnames`, `outputnames`

### Minimal Language Model (Auto-Detection Enabled)
```json
"my-gpt-model": {
    "max_length": 512,
    "summarization_task": "llm",
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "encoder_onnx_model": "model.onnx",
    "chunk_logic": "sentence",
    "encoder_only": true
}
```
**Auto-detected:** `dimension`, `vocab_size`, `inputnames`, `outputnames`

### Full Configuration (All Properties Specified)
```json
"my-complete-model": {
    "dimension": 768,
    "max_length": 512,
    "embedder_task": "fe",
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
        "mask": "attention_mask"
    },
    "outputnames": {
        "output": "last_hidden_state"
    },
    "encoder_onnx_model": "model.onnx",
    "use_optimized": true,
    "encoder_optimized_onnx_model": "model_optimized.onnx"
}
```

---

## 🔍 Auto-Detection Behavior

### Priority Order
1. **Config value** (if specified) - takes precedence
2. **Auto-detected value** (if config omitted) - extracted from ONNX model
3. **Default value** (if neither available) - system defaults

### Validation Rules
- **dimension:** If config > native, uses native (prevents upsampling)
- **vocab_size:** Must be > 1000 to be considered valid
- **inputnames:** Falls back to standard names (`input_ids`, `attention_mask`)
- **outputnames:** Uses first output if multiple outputs exist

### Logging
Auto-detection events are logged:
```
INFO: Auto-detected native dimension from ONNX model: 384
INFO: Auto-detected primary output name: last_hidden_state
DEBUG: Auto-detected output names from ONNX model: ['token_embeddings', 'sentence_embedding']
```

---

## 🚨 Special Cases

### T5 Models with Decoder Input
```json
"inputnames": {
    "use_decoder_input": true,
    "input": "input_ids",
    "mask": "attention_mask",
    "decoder_input_name": "decoder_input_ids"
}
```

### Models with Position IDs
```json
"inputnames": {
    "input": "input_ids",
    "mask": "attention_mask",
    "position": "position_ids"
}
```

### Seq2Seq Models with Separate Decoder Inputs
```json
"decoder_inputnames": {
    "encoder_output": "encoder_hidden_states",
    "input": "input_ids",
    "mask": "encoder_attention_mask"
}
```

---

## 📊 Property Reference Table

| Property | Auto-Detect? | Required? | Default | Applies To |
|----------|--------------|-----------|---------|------------|
| dimension | ✅ Yes | ❌ No | Auto | All models |
| inputnames | ✅ Yes | ❌ No | Auto | All models |
| outputnames | ✅ Yes | ❌ No | Auto | All models |
| vocab_size | ✅ Yes | ❌ No | 32000 | LLMs only |
| max_length | ❌ No | ✅ Yes | - | All models |
| embedder_task | ❌ No | ✅ Yes* | - | Embedding models |
| summarization_task | ❌ No | ✅ Yes* | - | Generation models |
| normalize | ❌ No | ✅ Yes | - | Embedding models |
| pooling_strategy | ❌ No | ✅ Yes | - | Embedding models |
| chunk_logic | ❌ No | ✅ Yes | - | All models |
| pad_token_id | ❌ No | ✅ Yes | - | Generation models |
| eos_token_id | ❌ No | ✅ Yes | - | Generation models |
| encoder_onnx_model | ❌ No | ✅ Yes | - | All models |
| chunk_overlap | ❌ No | ❌ No | 0 | All models |
| force_pooling | ❌ No | ❌ No | false | Embedding models |
| legacy_tokenizer | ❌ No | ❌ No | false | All models |
| lowercase | ❌ No | ❌ No | false | All models |
| remove_emojis | ❌ No | ❌ No | false | All models |
| use_optimized | ❌ No | ❌ No | false | All models |

*Either embedder_task OR summarization_task required, not both

---

## 💡 Best Practices

1. **Start Minimal:** Let auto-detection handle dimension, inputnames, outputnames, vocab_size
2. **Override When Needed:** Specify only if model behavior differs from auto-detected values
3. **Document Overrides:** Add inline notes explaining why you override auto-detection
4. **Test Auto-Detection:** Use the test script to verify auto-detected values:
   ```python
   python test_auto_detection.py
   ```
5. **Monitor Logs:** Check logs for auto-detection warnings and info messages

---

## 🔧 Troubleshooting

### Model not loading?
- Check `encoder_onnx_model` path is correct
- Verify ONNX model file exists in models directory

### Wrong dimension detected?
- Check model output shape: should be `[batch, seq, dim]`
- Specify dimension explicitly in config if auto-detection fails

### Invalid token IDs during generation?
- Ensure vocab_size is specified or auto-detected correctly
- Check pad_token_id, eos_token_id match tokenizer config

### Embeddings not normalized?
- Set `"normalize": true` explicitly in config
- This is required and cannot be auto-detected

---

**Last Updated:** December 30, 2025  
**Auto-Detection Features Introduced:** v2.0.0
