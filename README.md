> **Note:**  
> This project is under active development and we are looking for more collaborators to help improve and extend Flouds AI!  
> If you're interested in contributing, please reach out or open a pull request.

# Flouds AI

**Flouds AI** is a Python-based NLP service framework for text summarization and embedding, supporting both HuggingFace Transformers and ONNX runtime models. It is designed for extensibility, robust error handling, and easy integration into larger applications or microservices.

---

## Features

- **Text Summarization**: Supports both sequence-to-sequence language models and ONNX-based summarization with automatic sentence capitalization
- **Text Embedding**: Provides sentence and document embeddings using ONNX models with configurable chunking strategies
- **Batch Processing**: Efficiently handles batch summarization and embedding requests with async support
- **Model Optimization**: Supports both regular and optimized ONNX models with automatic fallback
- **Legacy Compatibility**: Handles tokenizer compatibility across different transformers versions
- **Configurable**: Per-model configuration with optimization flags, chunking logic, and tokenizer settings
- **Test Coverage**: Comprehensive unit tests with pytest and proper mocking
- **Production Ready**: Docker support, health checks, and environment-based configuration
- **FastAPI Powered**: Modern async API with automatic documentation

---

## Project Structure

```
app/
  config/
    appsettings.json        # Main application configuration
    onnx_config.json        # ONNX model configuration (see below)
  models/                   # Pydantic models for requests and responses
  modules/                  # Utility modules (e.g., concurrent dict)
  services/
    base_nlp_service.py     # Base class for NLP services
    embedder_service.py     # Embedding service logic
    summarizer_service.py   # Summarization service logic
  setup.py                  # App setup and environment preparation
onnx/                       # Default ONNX path (relative to working directory)
onnx_loaders/
  export_model.py           # Script for exporting HuggingFace models to ONNX
  load_scripts.txt          # Example commands for exporting models
tests/
  test_embedder_service.py
  test_summarizer_service.py
```

---

## Configuration

### appsettings.json

All main configuration is handled via `app/config/appsettings.json`.  
You can set server type, host, port, logging, ONNX options, and more.

**ONNX Model Path is Required:**  
You **must** set the ONNX model root path using the `onnx_path` field in the `onnx` section or override it with the `FLOUDS_ONNX_ROOT` environment variable.  
If not set, the application will exit with an error.

**Example:**
```json
{
    "app": {
        "name": "Flouds AI"
    },
    "server": {
        "type": "uvicorn",
        "host": "0.0.0.0",
        "port": 19690,
        "reload": false,
        "workers": 1,
        "session_provider": "CPUExecutionProvider"
    },
    "onnx": {
        "onnx_path": "onnx",
        "config_file": "onnx_config.json"
    }
}
```

- You can override any value using environment variables.

---

### onnx_config.json

The `onnx_config.json` file (located in `app/config/onnx_config.json`) is used to configure all ONNX models that you want to use in your application.  
Each entry in this file corresponds to a model you have downloaded and placed in your ONNX model folder.

- **Key**: The model name (e.g., `"t5-small"`, `"sentence-t5-base"`)
- **Value**: A dictionary describing the model's configuration, including:
  - `dimension`, `max_length`, `embedder_task` or `summarization_task`
  - `inputnames`, `outputnames`, `decoder_inputnames`
  - ONNX model file paths (`encoder_onnx_model`, `decoder_onnx_model`)
  - Optimized model paths (`encoder_optimized_onnx_model`, `decoder_optimized_onnx_model`)
  - Performance flags (`use_optimized`, `legacy_tokenizer`)
  - Special tokens, generation config, and other options

**Example snippet:**
```json
"t5-small": {
    "dimension": 512,
    "max_length": 512,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "summarization_task": "s2s",
    "legacy_tokenizer": true,
    "use_optimized": false,
    "inputnames": {
        "input": "input_ids",
        "mask": "attention_mask"
    },
    "outputnames": {
        "output": "last_hidden_state"
    },
    "decoder_inputnames": {
        "encoder_output": "encoder_hidden_states",
        "input": "input_ids",
        "mask": "encoder_attention_mask"
    },
    "encoder_onnx_model": "encoder_model.onnx",
    "decoder_onnx_model": "decoder_model.onnx",
    "encoder_optimized_onnx_model": "encoder_model_optimized.onnx",
    "decoder_optimized_onnx_model": "decoder_model_optimized.onnx",
    "special_tokens_map_path": "special_tokens_map.json",
    "num_beams": 4,
    "early_stopping": true,
    "use_seq2seqlm": false
}
```

- The structure of your ONNX model folder should match the configuration in this file.

#### ONNX Model Folder Structure

Model paths are organized by task name. For example, for summarization models with `summarization_task: "s2s"`, the path will be:

```
/onnx/models/s2s/t5-small/
    encoder_model.onnx
    decoder_model.onnx
    special_tokens_map.json
```

Here, `s2s` is the task name for summarization, and `t5-small` is the model name.  
For embedding models, the folder will use the `embedder_task` value.

**Note:**  
The `summarization_task` or `embedder_task` values are typically set to the same value as the `--model_for` argument used when exporting models to ONNX (see `onnx_loaders/load_scripts.txt`).  
Just make sure the path you specify in `summarization_task` or `embedder_task` matches the folder structure where your ONNX models and config files are stored.

#### `use_seq2seqlm` Option

- **`use_seq2seqlm: true`**  
  Uses `ORTModelForSeq2SeqLM` and its `.generate()` method for summarization (recommended for supported models).
- **`use_seq2seqlm: false`** (default)  
  Uses the lower-level `ort.InferenceSession` for both encoding and decoding.

#### Embedder Model Notes

- For embedding models, if your ONNX model file is named `model.onnx`, you do **not** need to specify the model name in the config (`encoder_onnx_model` is optional).
- If your model file has a different name, set `encoder_onnx_model` to the correct filename.
- The `logits` flag:  
  - If you know the encoder output is logits, set `"logits": true` in your config.
  - If not set, the process will try to detect if the output is logits and process accordingly.

---

## Development & Environment

- **FastAPI** is used as the main web framework for serving APIs.
- The server type (e.g., `uvicorn`) is set in `appsettings.json` and loaded dynamically.
- To run in development mode, set the following environment variables:
    - `FLOUDS_API_ENV=Development`
    - `FLOUDS_DEBUG_MODE=1`
- These variables control logging and debug behavior in the app.

Example (Windows CMD):
```cmd
set FLOUDS_API_ENV=Development
set FLOUDS_DEBUG_MODE=1
python -m app.main
```

Example (Linux/macOS):
```sh
export FLOUDS_API_ENV=Development
export FLOUDS_DEBUG_MODE=1
python -m app.main
```

---

## Exporting Models to ONNX

To use ONNX models, you need to export them from HuggingFace format.  
Use the scripts in `onnx_loaders/load_scripts.txt` as examples:

```plaintext
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/all-MiniLM-L6-v2" --optimize
python onnx_loaders/export_model.py --model_for "s2s" --model_name "t5-small" --optimize --task "seq2seq-lm"
python onnx_loaders/export_model.py --model_for "fe" --model_name "PleIAs/Pleias-Pico" --optimize
python onnx_loaders/export_model.py --model_for "s2s" --model_name "PleIAs/Pleias-Pico" --optimize --task "seq2seq-lm"
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/sentence-t5-base" --optimize --use_t5_encoder
python onnx_loaders/export_model.py --model_for "s2s" --model_name "google/pegasus-cnn_dailymail" --task "seq2seq-lm"
python onnx_loaders/export_model.py --model_for "s2s" --model_name "Falconsai/text_summarization" --task "seq2seq-lm" --model_folder "falconsai_text_summarization"
```

- `"fe"` = feature extraction (embedding)
- `"s2s"` = sequence-to-sequence (summarization)
- `--optimize` enables ONNX graph optimizations
- `--task` specifies the HuggingFace task type

---

## Usage

You can use the summarizer and embedder services as Python modules or integrate them into a web API.

### Example: Summarization

```python
from app.models.summarization_request import SummarizationRequest
from app.services.summarizer_service import TextSummarizer

req = SummarizationRequest(
    model="your-model-name",
    input="Your text to summarize.",
    temperature=0.7,  # Optional: controls randomness, between 0.0 and 2.0
)
response = TextSummarizer.summarize(req)
print(response.results.summary)
```

- **model**: Name of the model to use (e.g., `"t5-small"`).
- **input**: The text to be summarized.
- **temperature**: (Optional) Sampling temperature for generation (float, 0.0–2.0, default: 0.0).

#### Batch Summarization

```python
from app.models.summarization_request import SummarizationBatchRequest
from app.services.summarizer_service import TextSummarizer

batch_req = SummarizationBatchRequest(
    model="your-model-name",
    inputs=["Text 1 to summarize.", "Text 2 to summarize."],
    temperature=0.5,  # Optional
)
responses = TextSummarizer.summarize_batch(batch_req)
for resp in responses:
    print(resp.results.summary)
```

---

### Example: Embedding

```python
from app.models.embedding_request import EmbeddingRequest
from app.services.embedder_service import SentenceTransformer

req = EmbeddingRequest(
    model="your-model-name",
    input="Your text to embed.",
    projected_dimension=128,  # Optional, default: 128
)
response = SentenceTransformer.embed_text(
    text=req.input,
    model_to_use=req.model,
    projected_dimension=req.projected_dimension,
)
print(response.results)
```

- **model**: Name of the embedding model.
- **input**: The text to embed.
- **projected_dimension**: (Optional) Output embedding dimension (default: 128).

#### Batch Embedding

```python
from app.models.embedding_request import EmbeddingBatchRequest
from app.services.embedder_service import SentenceTransformer

batch_req = EmbeddingBatchRequest(
    model="your-model-name",
    inputs=["Text 1 to embed.", "Text 2 to embed."],
    projected_dimension=128,
)
response = SentenceTransformer.embed_batch_async(batch_req)
```

---

## Docker Usage

You can run Flouds AI as a Docker container for easy deployment.

### Docker Hub

Prebuilt images are available on [Docker Hub](https://hub.docker.com/r/gmalakar/flouds-ai-cpu):

```sh
docker pull gmalakar/flouds-ai-cpu:latest
```

### Build the Docker Image Locally

```sh
docker build -t flouds-ai-cpu .
```

For GPU support (requires CUDA drivers on host):

```sh
docker build --build-arg GPU=true -t flouds-ai-gpu .
```

### Start the Docker Container

```sh
docker run -p 19690:19690 \
  -v /path/to/your/onnx:/flouds-ai/onnx \
  -e FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
  -e FLOUDS_ONNX_CONFIG_FILE=/flouds-ai/onnx/onnx_config.json \
  gmalakar/flouds-ai-cpu
```

- `-v /path/to/your/onnx:/flouds-ai/onnx` mounts your ONNX model directory into the container.
- `-e FLOUDS_ONNX_ROOT` sets the ONNX model root path (**required**).
- `-e FLOUDS_ONNX_CONFIG_FILE` sets the ONNX config file path (optional, defaults to `onnx_config.json` in the model path).

---

### Using PowerShell Scripts for Docker

You can use the provided PowerShell scripts to build and run the Docker container on Windows:

#### Build the Docker Image

```powershell
.\build-flouds-ai.ps1
.\build-flouds-ai.ps1 -Tag v1.0.0
.\build-flouds-ai.ps1 -GPU
.\build-flouds-ai.ps1 -PushImage
```

#### Start the Docker Container

```powershell
.\start-flouds-ai.ps1
.\start-flouds-ai.ps1 -EnvFile .env -Tag v1.0.0
.\start-flouds-ai.ps1 -Force
```

- The script will validate required environment variables, map your ONNX and log directories, and set up Docker networks as needed.

**Tip:**  
You can also use the Bash script [`start-flouds-ai.sh`](start-flouds-ai.sh) for Linux/macOS environments.

---

## Significance of the `.env` File

The `.env` file allows you to set environment variables for configuration without modifying code.

**Common variables:**
- `FLOUDS_API_ENV` — Set to `Development` or `Production`
- `FLOUDS_DEBUG_MODE` — Set to `1` for debug logging
- `FLOUDS_ONNX_ROOT` — **(Required)** Path to your ONNX model directory
- `FLOUDS_ONNX_CONFIG_FILE` — Path to your ONNX config file (default: `onnx_config.json` in the model path)
- `FLOUDS_HOST` — Override server host (default: `0.0.0.0`)
- `FLOUDS_PORT` — Override server port (default: `19690`)

**Example `.env`:**
```
FLOUDS_API_ENV=Production
FLOUDS_DEBUG_MODE=0
FLOUDS_ONNX_ROOT=C:/path/to/your/onnx/models
FLOUDS_ONNX_CONFIG_FILE=C:/path/to/your/onnx_config.json
FLOUDS_HOST=0.0.0.0
FLOUDS_PORT=19690
```

You can mount your `.env` file into the container and load it using `python-dotenv`.

---

## Environment Variables

You can control the application behavior using the following environment variables:

- `FLOUDS_API_ENV` — Set to `Development` or `Production` (default: `Production`)
- `FLOUDS_DEBUG_MODE` — Set to `1` for debug logging, `0` for normal (default: `0`)
- `FLOUDS_ONNX_ROOT` — Path to the ONNX model root directory (**required**)
- `FLOUDS_ONNX_CONFIG_FILE` — Path to the ONNX config file (default: `onnx_config.json` in the model path)
- `FLOUDS_PORT` — Override the server port (default: `19690`)
- `FLOUDS_HOST` — Override the server host (default: `0.0.0.0`)
- `FLOUDS_SERVER_TYPE` — Server type, e.g., `uvicorn` or `hypercorn` (default: `uvicorn`)
- `FLOUDS_MODEL_SESSION_PROVIDER` — ONNX session provider (default: `CPUExecutionProvider`)

---

## ONNX Root Path

The ONNX model root directory is set in [`app/config/appsettings.json`](app/config/appsettings.json):

```json
"onnx": {
    "onnx_path": "onnx",
    "config_file": "onnx_config.json"
}
```

You can override this with the `FLOUDS_ONNX_ROOT` environment variable.

## Model Optimization

Flouds AI supports both regular and optimized ONNX models for better performance:

- **Regular models**: Direct ONNX conversion, slower but always compatible
- **Optimized models**: Graph-optimized versions, 20-50% faster inference
- **Per-model control**: Set `"use_optimized": true` in model config
- **Automatic fallback**: Uses regular models if optimized ones don't exist

## Legacy Tokenizer Support

For models exported with older transformers versions:

- Set `"legacy_tokenizer": true` in model config
- Automatically handles tokenizer compatibility issues
- Default is `false` for newly exported models

---

## FastAPI Routers

- **Summarizer endpoints** (see [`app/routers/summarizer.py`](app/routers/summarizer.py)):
  - `POST /summarize` — accepts a `SummarizationRequest`, returns a `SummarizationResponse`
  - `POST /summarize_batch` — accepts a `SummarizationBatchRequest`, returns a list of `SummarizationResponse`

- **Embedder endpoints** (see [`app/routers/embedder.py`](app/routers/embedder.py)):
  - `POST /embed` — accepts an `EmbeddingRequest`, returns an `EmbeddingResponse`
  - `POST /embed_batch` — accepts an `EmbeddingBatchRequest`, returns an `EmbeddingBatchResponse`

---

## Running Tests

Tests use `pytest` and extensive mocking for isolation.

```sh
pytest
```

---

## Contributing

1. Fork the repo and create your branch (`git checkout -b feature/your-feature`)
2. Commit your changes (`git commit -am 'Add new feature'`)
3. Push to the branch (`git push origin feature/your-feature`)
4. Create a new Pull Request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch](https://pytorch.org/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)

---

## Owner & Contact

**Owner:** Goutam Malakar  
**Contact:** Goutam Malakar

For questions or support, please open an issue or contact the owner.