# Flouds AI

**Flouds AI** is an enterprise-grade Python NLP service framework for text summarization and embedding, built with FastAPI and ONNX runtime. It features comprehensive monitoring, security, and performance optimizations for scalable deployment.

> **Note:** This project is under active development. If you're interested in contributing, please reach out or open a pull request.

## Key Features

### 🤖 AI Capabilities
- **Advanced Text Generation** – Seq2seq (T5, BART) and GPT-style models with ONNX optimization
- **Text Summarization** – Automatic sentence capitalization and advanced sampling strategies
- **High-Performance Embeddings** – Sentence and document embeddings with configurable chunking
- **File Content Extraction** – PDF, DOCX, PPT, Excel, TXT, HTML, CSV with structured output
- **Model Information API** – Real-time model availability, auto-detected parameters, configuration inspection
- **Batch Processing** – Async batch operations for high-throughput scenarios
- **Model Optimization** – ONNX models with automatic fallback and KV caching
- **Advanced Sampling** – Top-k, top-p, and repetition penalty for text generation

### 🚀 Enterprise Features
- **Performance Monitoring** – Real-time system metrics, memory tracking, performance profiling
- **Advanced Health Checks** – Component-based monitoring (ONNX, authentication, memory)
- **Enhanced Security** – CORS protection, log injection prevention, encrypted client authentication
- **Optimized Rate Limiting** – High-performance rate limiting with configurable thresholds
- **Resource Management** – Memory/CPU monitoring with automatic alerts
- **Docker Ready** – Multi-stage builds with GPU support and optimized images
- **Comprehensive Logging** – Structured logging with rotation, sanitization, configurable levels

## Overview

Professional AI inference server built on FastAPI for production deployment of transformers, embeddings, and language models with enterprise-grade monitoring and security.

## Quick Start

### Local Development

1. **Setup environment:**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Unix/Linux

pip install -r requirements-dev.txt
```

2. **Start the server:**
```bash
python -m app.main
```

Server runs on `http://localhost:8000` with API docs at `/docs`

### Docker Deployment

**Build and run:**
```bash
docker build -t flouds-ai .
docker run -p 8000:8000 \
  -e MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
  -e DEVICE=cuda \
  flouds-ai
```

**With GPU support:**
```bash
docker run -p 8000:8000 \
  --gpus all \
  -e DEVICE=cuda \
  flouds-ai
```

### PowerShell Deployment

**Start with automatic dependencies:**
```powershell
.\start-flouds-ai.ps1 -Model "sentence-transformers/all-MiniLM-L6-v2" -Device cuda -Port 8000
```

## Installation

1. Install Python 3.10+:
```bash
python --version
```

2. Install dependencies:
```bash
pip install -r app/requirements.txt
```

3. (Optional) Install development tools:
```bash
pip install -r requirements-dev.txt
```

## Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model identifier |
| `MODEL_TYPE` | `embedding` | Model type: `embedding`, `seq2seq`, `llm`, `classification` |
| `DEVICE` | `cpu` | Compute device: `cpu`, `cuda`, `mps` |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host binding |
| `WORKERS` | `4` | Number of worker threads |
| `DEBUG` | `False` | Enable debug logging |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_DIR` | `./logs` | Directory for log files |
| `FLOUDS_AUTH_ENABLED` | `True` | Enable token authentication |
| `FLOUDS_AUTH_TOKEN` | (required) | Bearer token for API access |
| `HF_TOKEN` | (optional) | HuggingFace API token for private models |
| `CHECKPOINT_PATH` | `./checkpoints/model` | Path to model checkpoints |

### Example .env File
```env
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
MODEL_TYPE=embedding
DEVICE=cuda
PORT=8000
HOST=0.0.0.0
WORKERS=4
DEBUG=False
LOG_LEVEL=INFO
FLOUDS_AUTH_ENABLED=True
FLOUDS_AUTH_TOKEN=your-secret-token-here
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

## API Endpoints

### Health & Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Basic health status |
| `GET` | `/health/detailed` | Detailed health info with memory, cache, disk stats |
| `GET` | `/health/ready` | Kubernetes readiness probe (ONNX path, auth check) |
| `GET` | `/health/live` | Kubernetes liveness probe |
| `GET` | `/health/performance` | Performance metrics and timing stats |
| `GET` | `/health/cache` | Cache statistics |
| `POST` | `/health/cache/clear` | Clear all caches |
| `POST` | `/health/cache/warmup` | Warm up model caches |

### Text Summarization
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/summarizer/summarize` | Summarize single text |
| `POST` | `/api/v1/summarizer/summarize_batch` | Summarize multiple texts in batch |

### Text Embedding
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/embedder/embed` | Generate embeddings for single text |
| `POST` | `/api/v1/embedder/embed_batch` | Generate embeddings for multiple texts |

### Text Extraction
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/extractor/extract` | Extract text from raw input |
| `POST` | `/api/v1/extractor/extract_file` | Extract text from uploaded file (PDF, DOCX, PPT, etc.) |

### Extract & Embed (Combined)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/extract-embed/extract_and_embed` | Extract and embed text in one request |
| `POST` | `/api/v1/extract-embed/extract_file_and_embed` | Extract text from file and generate embeddings |

### Prompt & Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/sendprompt/sendprompt` | Send prompt for processing |
| `POST` | `/api/v1/rag/generate` | Generate text with RAG (Retrieval-Augmented Generation) |

### Model Information
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/models/list` | List available models and their configurations |
| `POST` | `/api/v1/models/list` | List models with filters |

### Configuration Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/add` | Add new configuration |
| `GET` | `/api/v1/get` | Retrieve configuration |
| `PUT` | `/api/v1/update` | Update configuration |
| `DELETE` | `/api/v1/delete` | Delete configuration |

### Administration
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/generate-key` | Generate new API key for client |
| `DELETE` | `/api/v1/remove-client/{client_id}` | Remove client and revoke keys |
| `GET` | `/api/v1/clients` | List all registered clients |

## Project Structure

```
Flouds.Py/
├── app/
│   ├── config/
│   │   ├── appsettings.json              # Enterprise configuration
│   │   ├── appsettings.development.json  # Development overrides
│   │   ├── onnx_config.json             # ONNX model configurations
│   │   └── config_loader.py             # Configuration management
│   ├── middleware/
│   │   ├── rate_limit.py                # Advanced rate limiting
│   │   └── request_validation.py        # Request size/timeout validation
│   ├── models/                          # Pydantic request/response models
│   ├── routers/                         # FastAPI route handlers
│   ├── services/
│   │   ├── prompt_service.py            # Text generation and processing
│   │   ├── embedder_service.py          # Text embedding and similarity
│   │   ├── extractor_service.py         # File content extraction
│   │   └── base_nlp_service.py          # Shared NLP functionality
│   ├── utils/
│   │   ├── performance_monitor.py       # System performance tracking
│   │   ├── memory_monitor.py            # Memory usage monitoring
│   │   └── model_cache.py              # LRU model caching
│   ├── main.py                         # FastAPI application entry
│   ├── app_init.py                     # Initialization logic
│   ├── app_startup.py                  # Startup procedures
│   ├── healthcheck.py                  # Health check endpoints
│   ├── exceptions.py                   # Custom exceptions
│   ├── logger.py                       # Structured logging setup
│   └── __init__.py
├── checkpoints/                         # Model checkpoint storage
├── data/                                # Training/validation data
├── docs/                                # Documentation
├── logs/                                # Application logs
├── tests/                               # Test suite
├── Dockerfile                           # Container image definition
├── docker-compose.yml                   # Multi-service orchestration
├── requirements.txt                     # Production dependencies
├── requirements-dev.txt                 # Development dependencies
├── pyproject.toml                       # Project metadata
├── pytest.ini                           # Pytest configuration
├── .env.example                         # Environment template
└── README.md                            # This file
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Embedding Process Flow](docs/EMBEDDING.md)** – Detailed embedding pipeline with pooling, projection, normalization, and quantization
- **[Model Information API](docs/MODEL_INFO_API.md)** – Model availability checks and auto-detected parameters
- **[Environment Variables](docs/ENVIRONMENT.md)** – Complete environment variables reference
- **[Cache Keys Utility](docs/CACHE_KEYS.md)** – Cache key canonicalization and best practices

## Performance & Optimization

### Model Loading
- Models are loaded at startup and cached in memory
- GPU models benefit from mixed-precision inference (FP16)
- Batch inference provides better throughput than single requests

### Request Handling
- Async request processing for concurrent API calls
- Connection pooling for database/remote model access
- Request timeouts prevent hanging operations

### Monitoring
- Prometheus metrics exposed at `/metrics`
- Request latency tracking
- Model inference time profiling
- Memory usage monitoring

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | `pip install -r app/requirements.txt` |
| `CUDA out of memory` | Reduce batch size, use FP16 mode, or switch to CPU |
| `Model not found on HuggingFace` | Verify model name, check HF token, check internet connectivity |
| `401 Unauthorized` on API calls | Verify `FLOUDS_AUTH_TOKEN` in header matches server configuration |
| `Connection refused: localhost:8000` | Check if server is running and PORT is correct |
| `Slow inference times` | Enable CUDA if available, check for CPU bottleneck with `top`/Task Manager |

## Development

### Run Tests
```bash
pytest tests/ -v
pytest tests/ -v --cov=app  # with coverage
```

### Code Quality
```bash
black app/                    # Format code
isort app/                    # Sort imports
pylint app/                   # Linting
mypy app/                     # Type checking
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Logging

Logs are saved to `logs/` directory and also printed to console. Log levels:

- **DEBUG** – Detailed diagnostic information
- **INFO** – General informational messages
- **WARNING** – Warning messages for potentially problematic situations
- **ERROR** – Error messages for failures

Configure log level via `LOG_LEVEL` environment variable.

## Security

- **Token Authentication** – Required bearer token for all API endpoints (configurable)
- **CORS** – Configure allowed origins in `app/middleware/`
- **Rate Limiting** – Implement via middleware for production
- **HTTPS** – Use reverse proxy (nginx/traefik) for production TLS
- **Environment Secrets** – Store sensitive data in `.env` (never commit)

## Requirements

- **Python**: 3.10 or later
- **Dependencies**: See `app/requirements.txt`
- **GPU** (optional): CUDA 11.8+ for GPU acceleration
- **System**: ≥4 GB RAM (8+ GB recommended for large models)

## Performance Characteristics

| Model | Device | Avg Latency | Throughput |
|-------|--------|-------------|-----------|
| all-MiniLM-L6-v2 (90M) | CPU | 50-100ms | 10-20 req/s |
| all-MiniLM-L6-v2 (90M) | CUDA | 5-10ms | 100-200 req/s |
| deepseek-coder-1.3b (1.3B) | CUDA | 200-500ms | 2-5 req/s |
| Phi-3-mini (3.8B) | CUDA | 500-1000ms | 1-2 req/s |

*Performance varies based on batch size, sequence length, and hardware.*

## Deployment

### Production Checklist
- [ ] Set `DEBUG=False`
- [ ] Generate strong `FLOUDS_AUTH_TOKEN`
- [ ] Configure `CHECKPOINT_PATH` for model persistence
- [ ] Enable HTTPS via reverse proxy
- [ ] Setup log rotation
- [ ] Configure resource limits (memory, CPU)
- [ ] Monitor `/metrics` endpoint
- [ ] Setup alerting for `/health` failures
- [ ] Test graceful shutdown

### Scaling Options
- **Horizontal** – Run multiple containers behind load balancer
- **Vertical** – Increase CPU/RAM, enable CUDA for GPU acceleration
- **Caching** – Implement Redis for response caching
- **Rate Limiting** – Add middleware to protect against overload

## License

See LICENSE file.

## Contributing

See CONTRIBUTING.md for guidelines.
