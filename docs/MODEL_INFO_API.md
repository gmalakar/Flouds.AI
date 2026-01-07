````markdown
# Model Information API Documentation

## Overview

The Model Information API provides comprehensive details about ONNX models including availability checks, auto-detected parameters, configuration values, and file information.

## Endpoints

### 1. Get Model Information

**Endpoint:** `GET /api/v1/model/info`

**Description:** Retrieves detailed information about a specific model including auto-detected parameters from ONNX files.

**Query Parameters:**
- `model` (string, required): Name of the model to check

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
  }
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

## Auto-Detected Parameters

The API automatically detects the following parameters from ONNX model files:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `dimension` | Output embedding dimension | 384 |
| `inputnames` | Model input tensors | `input_ids`, `attention_mask` |
| `outputnames` | Model output tensors | `last_hidden_state` |
````
