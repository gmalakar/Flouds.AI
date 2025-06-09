> **Note:**  
> This project is under active development and we are looking for more collaborators to help improve and extend Flouds.Py!  
> If you're interested in contributing, please reach out or open a pull request.

# Flouds.Py

**Flouds.Py** is a Python-based NLP service framework for text summarization and embedding, supporting both HuggingFace Transformers and ONNX runtime models. It is designed for extensibility, robust error handling, and easy integration into larger applications or microservices.

---

## Features

- **Text Summarization**: Supports both sequence-to-sequence language models and ONNX-based summarization.
- **Text Embedding**: Provides sentence and document embeddings using ONNX or HuggingFace models.
- **Batch Processing**: Efficiently handles batch summarization and embedding requests.
- **Configurable**: Easily switch between models and runtime backends via configuration.
- **Test Coverage**: Includes comprehensive unit tests with pytest and mocking for all major service paths.
- **Extensible**: Designed for easy extension to new NLP tasks and models.
- **FastAPI Powered**: Uses [FastAPI](https://fastapi.tiangolo.com/) for serving APIs.

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

Example:
```json
{
    "app": {
        "name": "Flouds PY"
    },
    "server": {
        "type": "uvicorn",
        "host": "0.0.0.0",
        "port": 5001,
        "reload": true,
        "workers": 4,
        "model_session_provider": "CPUExecutionProvider"
    },
    "onnx": {
        "config_check_interval": 10
    },
    "logging": {
        "folder": "logs",
        "app_log_file": "flouds.log"
    }
}
```

---

### onnx_config.json

The `onnx_config.json` file (located in `app/config/onnx_config.json`) is used to configure all ONNX models that you want to use in your application.  
Each entry in this file corresponds to a model you have downloaded and placed in your ONNX model folder.

- **Key**: The model name (e.g., `"t5-small"`, `"sentence-t5-base"`)
- **Value**: A dictionary describing the model's configuration, including:
  - `dimension`, `max_length`, `embedder_task` or `summarization_task`
  - `inputnames`, `outputnames`, `decoder_inputnames`
  - ONNX model file paths (`encoder_onnx_model`, `decoder_onnx_model`)
  - Special tokens, generation config, and other options

**Example snippet:**
```json
"t5-small": {
    "dimension": 512,
    "max_length": 512,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "logits": false,
    "summarization_task": "s2s",
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
    "decoder_onnx_model": "decoder_model.onnx",
    "encoder_onnx_model": "encoder_model.onnx",
    "special_tokens_map_path": "special_tokens_map.json",
    "num_beams": 4,
    "early_stopping": true,
    "use_seq2seqlm": false
}
```

**How it works:**
- You must add an entry for each ONNX model you want to use.
- The paths and configuration options are defined according to the `embedder_task` (for embedding models) or `summarization_task` (for summarization models).
- The structure of your ONNX model folder should match the configuration in this file.

#### ONNX Model Folder Structure

Model paths are organized by task name. For example, for summarization models with `summarization_task: "s2s"`, the path will be:

```
app/onnx/models/s2s/t5-small/
    encoder_model.onnx
    decoder_model.onnx
    special_tokens_map.json
```

Here, `s2s` is the task name for summarization, and `t5-small` is the model name.  
For embedding models, the folder will use the `embedder_task` value.

**Note:**  
The `summarization_task` or `embedder_task` values are typically set to the same value as the `--model_for` argument used when exporting models to ONNX (see `onnx_loaders/load_scripts.txt`).  
However, you are free to use any folder name for your task (e.g., `s2s`, `fe`, or a custom name) and organize your ONNX models and configurations accordingly.  
Just make sure the path you specify in `summarization_task` or `embedder_task` matches the folder structure where your ONNX models and config files are stored.

If you want to use the same ONNX model for both summarization and embedding tasks, you can specify the same path for both `summarization_task` and `embedder_task` in your configuration.

#### `use_seq2seqlm` Option

There are two ways to perform summarization with ONNX models:

- **`use_seq2seqlm: true`**  
  If you set `use_seq2seqlm` to `true`, the service will use `ORTModelForSeq2SeqLM` and its `.generate()` method for summarization, which closely mimics HuggingFace's generate pipeline and is typically easier to use for supported models.

- **`use_seq2seqlm: false`** (default)  
  If you set `use_seq2seqlm` to `false` (or omit it), the service will use the lower-level `ort.InferenceSession` for both encoding and decoding, giving you more control and compatibility with custom ONNX models.

Choose the option that matches your exported ONNX model and desired inference flow.

#### Embedder Model Notes

- For embedding models, if your ONNX model file is named `model.onnx`, you do **not** need to specify the model name in the config (`encoder_onnx_model` is optional).
- If your model file has a different name, set `encoder_onnx_model` to the correct filename (see the `sentence-t5-base` example).
- The `logits` flag:  
  - If you know the encoder output is logits, set `"logits": true` in your config.
  - If not set (default is `false`), the process will try to detect if the output is logits and process accordingly.
- For embedders, ONNX `InferenceSession` will always be used.

---

## Development & Environment

- **FastAPI** is used as the main web framework for serving APIs.
- The server type (e.g., `uvicorn`) is set in `appsettings.json` and loaded dynamically.
- To run in development mode, set the following environment variables:
    - `FASTAPI_ENV=Development`
    - `FASTAPI_DEBUG_MODE=1`
- These variables control logging and debug behavior in the app.

Example (Windows CMD):
```cmd
set FASTAPI_ENV=Development
set FASTAPI_DEBUG_MODE=1
python -m app.main
```

Example (Linux/macOS):
```sh
export FASTAPI_ENV=Development
export FASTAPI_DEBUG_MODE=1
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

**Note:**  
The `summarization_task` or `embedder_task` in your `onnx_config.json` are typically set to the same value as the `--model_for` argument used when exporting models to ONNX.  
However, you can use any name for your task folder and keep your ONNX models and configurations organized as you prefer.  
Just ensure the path you specify in `summarization_task` or `embedder_task` matches your folder structure.  
If you want to use the same ONNX model for both tasks, you can specify the same path for both.

See `onnx_loaders/load_scripts.txt` for more examples.

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

### FastAPI Routers

- **Summarizer endpoints** (see [`app/routers/summarizer.py`](app/routers/summarizer.py)):
  - `POST /summarize` — accepts a `SummarizationRequest`, returns a `SummarizationResponse`
  - `POST /summarize_batch` — accepts a `SummarizationBatchRequest`, returns a list of `SummarizationResponse`

- **Embedder endpoints** (see [`app/routers/embedder.py`](app/routers/embedder.py)):
  - `POST /embed` — accepts an `EmbeddingRequest`, returns an `EmbeddingResponse`
  - `POST /embed_batch` — accepts an `EmbeddingBatchRequest`, returns an `EmbeddingBatchResponse`

See the routers for full request/response models and details.

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
