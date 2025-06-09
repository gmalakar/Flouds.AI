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
python onnx_loaders/export_model.py --model_for "s2s" --model_name "facebook/bart-large-cnn" --task "text2text-generation" --use_cache
```

- `"fe"` = feature extraction (embedding)
- `"s2s"` = sequence-to-sequence (summarization)
- `--optimize` enables ONNX graph optimizations
- `--task` specifies the HuggingFace task type

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
    use_optimized_model=False
)
response = TextSummarizer.summarize(req)
print(response.results.summary)
```

### Example: Embedding

```python
from app.services.embedder_service import SentenceTransformer

embedding = SentenceTransformer.embed_text(
    "Your text to embed.",
    model_name="your-model-name"
)
print(embedding)
```

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
