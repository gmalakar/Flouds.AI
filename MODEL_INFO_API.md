# Model Information API Documentation

## Overview

The Model Information API provides comprehensive details about ONNX models including availability checks, auto-detected parameters, configuration values, and file information.

## Endpoints

### 1. Get Model Information

**Endpoint:** `GET /api/v1/model/info`

**Description:** Retrieves detailed information about a specific model including auto-detected parameters from ONNX files.

**Query Parameters:**
- `model` (string, required): Name of the model to check

**Response Model:** `ModelInfoResponse` (extends `BaseResponse`)

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the operation was successful |
| `message` | string | Status message |
| `model` | string | Model name queried |
| `time_taken` | float | Time taken for the operation |
| `warnings` | List[string] | Warning messages (e.g., missing files) |
| `model_available` | boolean | Whether model exists in configuration |
| `onnx_file_available` | boolean | Whether ONNX files exist on disk |
| `model_type` | string | Type: 'embedding', 'summarization', or 'language_model' |
| `auto_detected_params` | object | Parameters auto-detected from ONNX model |
| `config_params` | object | Configuration parameters (filtered by model type) |
| `default_params` | object | Default values for this model type |
| `required_params` | List[string] | List of required parameter names |
| `files_info` | object | Information about model files |

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/model/info?model=all-MiniLM-L6-v2"
```

**Example Response:**

```json
{
  "success": true,
  "message": "Model information retrieved successfully",
  "model": "all-MiniLM-L6-v2",
  "time_taken": 0.52,
  "warnings": [],
  "model_available": true,
  "onnx_file_available": true,
  "model_type": "embedding",
  "auto_detected_params": {
    "dimension": 384,
    "outputnames": ["token_embeddings", "sentence_embedding"],
    "primary_output": "token_embeddings",
    "inputnames": ["input_ids", "attention_mask"]
  },
  "config_params": {
    "max_length": 256,
    "embedder_task": "fe",
    "normalize": true,
    "pooling_strategy": "mean",
    "chunk_logic": "sentence",
    "chunk_overlap": 1,
    "force_pooling": true,
    "encoder_onnx_model": "model.onnx",
    "use_optimized": true,
    "encoder_optimized_onnx_model": "model_optimized.onnx"
  },
  "default_params": {
    "legacy_tokenizer": false,
    "lowercase": false,
    "remove_emojis": false,
    "force_pooling": false,
    "chunk_overlap": 0,
    "use_optimized": false
  },
  "required_params": [
    "max_length",
    "chunk_logic",
    "encoder_onnx_model",
    "embedder_task",
    "normalize",
    "pooling_strategy"
  ],
  "files_info": {
    "encoder_file": "model.onnx",
    "encoder_exists": true,
    "encoder_optimized_file": "model_optimized.onnx",
    "encoder_optimized_exists": true,
    "decoder_file": null,
    "decoder_exists": false,
    "decoder_optimized_file": null,
    "decoder_optimized_exists": false
  }
}
```

### 2. List All Models

**Endpoint:** `GET /api/v1/models/list`

**Description:** Lists all available models in the configuration.

**Response:**

```json
{
  "success": true,
  "message": "Found 10 models",
  "models": [
    {
      "name": "all-MiniLM-L6-v2",
      "type": "embedding"
    },
    {
      "name": "distilgpt2",
      "type": "language_model"
    },
    {
      "name": "t5-small",
      "type": "summarization"
    }
  ]
}
```

## Use Cases

### 1. Check Model Availability Before Use

```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/model/info",
    params={"model": "all-MiniLM-L6-v2"}
)
data = response.json()

if data["model_available"] and data["onnx_file_available"]:
    print("Model is ready to use!")
    print(f"Dimension: {data['auto_detected_params']['dimension']}")
else:
    print("Model not available")
```

### 2. Validate Model Configuration

```python
response = requests.get(
    "http://localhost:8000/api/v1/model/info",
    params={"model": "my-custom-model"}
)
data = response.json()

# Check if all required parameters are configured
required = data["required_params"]
config = data["config_params"]

missing = [param for param in required if param not in config]
if missing:
    print(f"Missing required parameters: {missing}")
```

### 3. Get Auto-Detected Values

```python
response = requests.get(
    "http://localhost:8000/api/v1/model/info",
    params={"model": "all-MiniLM-L6-v2"}
)
data = response.json()

auto_detected = data["auto_detected_params"]
print(f"Native dimension: {auto_detected['dimension']}")
print(f"Input names: {auto_detected['inputnames']}")
print(f"Output names: {auto_detected['outputnames']}")
```

### 4. List All Available Models

```python
response = requests.get("http://localhost:8000/api/v1/models/list")
data = response.json()

embedding_models = [m for m in data["models"] if m["type"] == "embedding"]
print(f"Available embedding models: {[m['name'] for m in embedding_models]}")
```

## Auto-Detected Parameters

The API automatically detects the following parameters from ONNX model files:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `dimension` | Embedding dimension from output shape | 384, 768, 512 |
| `inputnames` | Required input tensor names | ["input_ids", "attention_mask"] |
| `outputnames` | Output tensor names | ["token_embeddings", "sentence_embedding"] |
| `primary_output` | Primary output tensor name | "token_embeddings" |
| `vocab_size` | Vocabulary size (language models) | 50257 |

## Model Types

| Type | Code | Description |
|------|------|-------------|
| `embedding` | `fe` | Feature extraction / text embedding models |
| `summarization` | `s2s` | Sequence-to-sequence summarization models |
| `language_model` | `llm` | Large language models for text generation |

## Error Responses

### Model Not Found in Configuration

```json
{
  "success": false,
  "message": "Model 'unknown-model' not found in configuration",
  "model": "unknown-model",
  "model_available": false,
  "onnx_file_available": false
}
```

### ONNX Files Missing

```json
{
  "success": true,
  "message": "Model found in config but ONNX files are missing",
  "model": "my-model",
  "model_available": true,
  "onnx_file_available": false,
  "warnings": [
    "ONNX model file 'model.onnx' not found"
  ]
}
```

## Key Features

### 1. No Internal Paths Exposed
- Model file paths are kept internal for security
- Only file existence status is returned

### 2. Type-Specific Parameter Filtering
- **Embedding models** receive only embedding-related parameters
  - Example: `pooling_strategy`, `normalize`, `force_pooling`
  - Excludes generation parameters like `temperature`, `num_beams`
  
- **Summarization/Language models** receive only generation-related parameters
  - Example: `num_beams`, `temperature`, `top_k`, `do_sample`
  - Excludes embedding parameters like `pooling_strategy`, `normalize`

### 3. Optimized Default Parameters
- Only defaults relevant to the model type are returned
- Reduces response size and improves clarity

## Integration with Swagger UI

Access the interactive API documentation at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

The Model Information endpoints are available under the "Model Information" tag.

## Testing

Run the test script:

```bash
python test_model_info_api.py
```

Or test manually with curl:

```bash
# Get model info
curl -X GET "http://localhost:8000/api/v1/model/info?model=all-MiniLM-L6-v2"

# List all models
curl -X GET "http://localhost:8000/api/v1/models/list"
```

## Notes

- **Security:** Internal file paths are not exposed in responses
- **Filtering:** Parameters are filtered by model type to reduce response size
- **Auto-detection:** Requires ONNX files to be present on disk
- **Configuration:** Parameters are always returned, even if files are missing
- **Performance:** Response times typically 0.3-1.0 seconds depending on model size
- **Validation:** Respects the same path validation and security measures as other endpoints

---

**Version:** 1.1.0  
**Last Updated:** December 30, 2025
