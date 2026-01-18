# Flouds AI

> **Production-ready NLP service framework** for text summarization and embedding, built with FastAPI and ONNX Runtime.
>
> Designed for enterprise deployment with comprehensive observability, security hardening, and performance optimizations.

[![Tests](https://img.shields.io/badge/tests-195%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Enterprise features**: Structured logging • Multi-tenant isolation • OWASP security • Performance monitoring • Docker ready

**Latest Release (January 2026):**
- ✅ Structured JSON logging with request correlation and sanitization
- ✅ Embedder service refactored into 8 modular components
- ✅ Prompt service refactored into 7 specialized modules
- ✅ 195 comprehensive tests with 100% pass rate

---

## Table of Contents

- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Quick Start](#-quick-start)
- [Testing](#-testing)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [Security](#-security-features)
- [Observability](#-observability)
- [Docker Deployment](#-docker-deployment)
- [Production Deployment](#-production-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Key Features

### 🤖 **AI & NLP Capabilities**
- **Advanced Text Generation**: Seq2seq (T5, BART, Pegasus) and GPT-style models optimized with ONNX Runtime
- **Text Summarization**: Production-quality summarization with automatic capitalization and advanced sampling (top-k, top-p, repetition penalty)
- **High-Performance Embeddings**: Sentence/document embeddings with:
  - Configurable chunking strategies and token limits
  - Dimension projection with deterministic caching
  - Multiple quantization options (int8, uint8, binary) for storage optimization
  - Batch processing with async support
- **File Content Extraction**: Extract text from PDF, DOCX, DOC, PPTX, XLSX, TXT, HTML, CSV with structured output
- **Model Auto-Detection**: Automatic detection of dimensions, input/output names, and vocab_size from ONNX models
- **Multi-Level Caching**: Model cache, embedding cache, generation cache, and projection matrix cache for optimal performance

### 🔒 **Security & Compliance**
- **OWASP-Compliant Headers**: X-Frame-Options, X-Content-Type-Options, CSP, HSTS, Referrer-Policy
- **Path Traversal Protection**: Comprehensive validation with safe file operations
- **Log Injection Prevention**: Automatic sanitization of sensitive data in logs
- **Request Size Limits**: Configurable payload limits (10MB default) with 413 responses
- **Encrypted Client Authentication**: Secure API key management with strict file permissions (0600)
- **Tenant Isolation**: Multi-tenant support with trusted host validation

### 📊 **Observability & Monitoring**
- **Structured JSON Logging**: Production-ready logging with:
  - Request correlation via X-Request-ID headers
  - Automatic timing and duration tracking (microsecond precision)
  - Endpoint metadata (HTTP method, path, status code)
  - Safe payload sanitization with sensitive key redaction
  - Context variables for tenant, user, and request tracking
- **Performance Monitoring**: Real-time CPU, memory, and disk metrics with threshold alerts
- **Health Checks**: Component-based health monitoring (ONNX, auth, memory, cache)
- **Rate Limiting**: Optimized rate limiting with efficient timestamp management and batched cleanup

### ⚙️ **Architecture & Design**
- **Modular Services**: Refactored architecture with:
  - **Prompt Service**: 7 specialized modules (models, text_utils, onnx_utils, decoder, encoder, generation, prompt)
  - **Embedder Service**: 8 focused modules (models, text_utils, onnx_utils, processing, resource_manager, inference, embedder, core)
- **Async-First**: Full async/await support with configurable thread pools
- **Environment-Aware Config**: Development/production configs with environment variable overrides
- **Docker Ready**: Multi-stage builds with GPU support and optimized images
- **Comprehensive Testing**: 195 tests covering unit, integration, and security scenarios

---

## 📁 Project Structure

```
app/
  ├── config/
  │   ├── appsettings.json              # Base configuration
  │   ├── appsettings.development.json  # Development overrides
  │   ├── appsettings.production.json   # Production settings
  │   ├── onnx_config.json              # ONNX model configurations
  │   └── config_loader.py              # Configuration management
  │
  ├── middleware/
  │   ├── rate_limit.py                 # Optimized rate limiting
  │   ├── request_validation.py         # Size/timeout validation
  │   ├── log_context.py                # Request context & timing
  │   ├── tenant_security.py            # Multi-tenant isolation
  │   └── auth_middleware.py            # Authentication enforcement
  │
  ├── services/
  │   ├── prompt/                       # Text generation (7 modules)
  │   │   ├── models.py                 # Pydantic request/response models
  │   │   ├── text_utils.py             # Text preprocessing utilities
  │   │   ├── onnx_utils.py             # ONNX session management
  │   │   ├── decoder.py                # Decoder-only model logic
  │   │   ├── encoder.py                # Encoder operations
  │   │   ├── generation.py             # Token generation & sampling
  │   │   └── prompt.py                 # Main service facade
  │   │
  │   ├── embedder/                     # Embeddings (8 modules)
  │   │   ├── models.py                 # Request/response schemas
  │   │   ├── text_utils.py             # Text chunking & preprocessing
  │   │   ├── onnx_utils.py             # ONNX inference utilities
  │   │   ├── processing.py             # Projection & quantization
  │   │   ├── resource_manager.py       # Session/tokenizer management
  │   │   ├── inference.py              # Embedding computation
  │   │   ├── embedder.py               # Main embedder service
  │   │   └── __init__.py               # Module exports
  │   │
  │   ├── embedder_service.py           # Backward-compatible wrapper
  │   ├── extractor_service.py          # File extraction service
  │   └── base_nlp_service.py           # Shared NLP functionality
  │
  ├── routers/                          # FastAPI route handlers
  ├── models/                           # Pydantic models
  ├── utils/                            # Performance & monitoring utilities
  ├── logger.py                         # Structured JSON logging
  └── main.py                           # FastAPI application

onnx_loaders/                           # Model export utilities
tests/                                  # 195 comprehensive tests
docs/                                   # Documentation
```

---

## 📚 Documentation

### Getting Started
- **[Setup & Installation](docs/SETUP.md)** - Installation, configuration, and getting started guide
- **[Environment Variables](docs/ENVIRONMENT.md)** - Complete environment variables reference
- **[Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md)** - Pre-deployment and post-deployment verification

### Architecture & Design
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System architecture, data flows, and security model
- **[Embedding Process Flow](docs/EMBEDDING.md)** - Detailed embedding pipeline with pooling, projection, normalization, quantization
- **[Cache Keys Utility](docs/CACHE_KEYS.md)** - Cache key canonicalization and best practices

### API & Features
- **[Model Information API](docs/MODEL_INFO_API.md)** - Model availability checks and auto-detected parameters
- **[Structured Logging](docs/STRUCTURED_LOGGING.md)** - JSON logging, request correlation, and observability

### Security & Maintenance
- **[Security Fixes](docs/SECURITY_FIX_SUMMARY.md)** - Security improvements and hardening measures
- **[Requirements Organization](docs/REQUIREMENTS_REORGANIZATION.md)** - Dependency management and organization
- **[Requirements Analysis](docs/REQUIREMENTS_ANALYSIS.md)** - Development vs production dependency verification

### Development
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project
- **[Refactoring Summary](docs/PHASE_B_COMPLETION_REPORT.md)** - Phase B refactoring details and metrics

---

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.10 or higher
- **ONNX Runtime**: 1.14+ (CPU or GPU)
- **System**: Linux, macOS, or Windows

### Installation

```bash
# Clone repository (replace with your fork)
git clone https://github.com/YOUR_USERNAME/Flouds.git
cd Flouds.Py

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Install production dependencies
pip install -r app/requirements-prod.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set required variables:
# - FLOUDS_ONNX_ROOT: Path to ONNX models directory
# - FLOUDS_ONNX_CONFIG_FILE: Path to onnx_config.json
# - FLOUDS_API_ENV: "Development" or "Production"
```

### Running the Server

```bash
# Development mode (with auto-reload and debug logging)
export FLOUDS_API_ENV="Development"
export FLOUDS_LOG_JSON="1"  # Enable structured JSON logging
python -m app.main

# Production mode (optimized performance)
export FLOUDS_API_ENV="Production"
export FLOUDS_ONNX_ROOT="/path/to/models"
export FLOUDS_ONNX_CONFIG_FILE="/path/to/onnx_config.json"
uvicorn app.main:app --host 0.0.0.0 --port 19690 --workers 4
```

### Using Docker

```bash
# Build image
docker build -t flouds-ai:latest .

# Run container
docker run -d \
  --name flouds-ai \
  -p 19690:19690 \
  -e FLOUDS_API_ENV=Production \
  -e FLOUDS_ONNX_ROOT=/models \
  -e FLOUDS_LOG_JSON=1 \
  -v /path/to/models:/models:ro \
  -v /path/to/logs:/app/logs \
  flouds-ai:latest

# Or use docker-compose
docker-compose up -d
```

### API Documentation

Once running, access interactive documentation:
- **Swagger UI**: http://localhost:19690/api/v1/docs
- **ReDoc**: http://localhost:19690/api/v1/redoc
- **OpenAPI Spec**: http://localhost:19690/api/v1/openapi.json
- **Health Check**: http://localhost:19690/api/v1/health

---

## 🧪 Testing

```bash
# Run all tests (195 tests)
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_embedder_service.py -v
pytest tests/test_quantization.py -v
pytest tests/test_structured_logging.py -v

# Run tests matching pattern
pytest -k "security" -v

# Run with output capturing disabled (useful for debugging)
pytest -s tests/test_prompt_service.py
```

**Test Coverage:**
- ✅ Embedder service: 44 tests (quantization, projection, caching, text processing)
- ✅ Prompt service: 8 tests (generation, caching, sampling)
- ✅ Security: 15 tests (path validation, headers, size limits, auth)
- ✅ Middleware: 6 tests (rate limiting, logging, context)
- ✅ Configuration: 8 tests (caching, loading, validation)
- ✅ Integration: 114 additional tests covering extractors, models, utilities

---

## 📦 API Endpoints

### Text Generation & Summarization

**Summarize Text**
```http
POST /api/v1/summarizer/summarize
Content-Type: application/json

{
  "text": "Long text to summarize...",
  "max_length": 150,
  "min_length": 50,
  "model_name": "t5-small",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

### Text Embeddings

**Generate Embeddings**
```http
POST /api/v1/embedder/embed
Content-Type: application/json

{
  "texts": ["First sentence", "Second sentence"],
  "model_name": "sentence-transformers",
  "normalize": true,
  "projected_dimension": 512,
  "quantize": "int8"
}
```

**Batch Embeddings**
```http
POST /api/v1/embedder/embed/batch
Content-Type: application/json

{
  "texts": ["text1", "text2", "..."],
  "batch_size": 32
}
```

**Compute Similarity**
```http
GET /api/v1/embedder/similarity?text1=hello&text2=hi
```

### File Extraction

**Extract Text from File**
```http
POST /api/v1/extractor/extract
Content-Type: multipart/form-data

file: <binary file data>
```

**Extract and Embed**
```http
POST /api/v1/extractor/extract-and-embed
Content-Type: multipart/form-data

file: <binary file data>
normalize: true
model_name: sentence-transformers
```

### Model Information

**List All Models**
```http
GET /api/v1/models
```

**Get Model Details**
```http
GET /api/v1/models/t5-small
```

### Health & Monitoring

**Health Check**
```http
GET /api/v1/health
```

Response includes:
- Overall status and timestamp
- ONNX Runtime availability
- Authentication system status
- Memory usage and thresholds
- Cache statistics

---

## 🔧 Configuration

Configuration follows a hierarchical override system:

1. **Base Config**: `app/config/appsettings.json`
2. **Environment-Specific**: `app/config/appsettings.{FLOUDS_API_ENV}.json`
3. **Environment Variables**: `FLOUDS_*` prefixed variables (highest priority)
4. **Runtime Overrides**: Command-line arguments and programmatic changes

### Key Configuration Options

**Logging**
```bash
FLOUDS_LOG_JSON=1                    # Enable structured JSON logging
FLOUDS_LOG_LEVEL=INFO               # DEBUG, INFO, WARNING, ERROR, CRITICAL
FLOUDS_LOG_FILE=logs/flouds.log     # Log file path
```

**Performance**
```bash
FLOUDS_CACHE_MAX_SIZE=1000          # Max cache entries
FLOUDS_CACHE_TTL=3600               # Cache TTL in seconds
FLOUDS_THREAD_POOL_SIZE=4           # Thread pool workers
```

**Security**
```bash
FLOUDS_REQUEST_SIZE_LIMIT=10485760  # 10MB request limit
FLOUDS_RATE_LIMIT_REQUESTS=100      # Requests per window
FLOUDS_RATE_LIMIT_WINDOW=60         # Window size in seconds
```

**Models**
```bash
FLOUDS_ONNX_ROOT=/path/to/models    # ONNX models directory
FLOUDS_ONNX_CONFIG_FILE=onnx_config.json  # Model configuration
```

See [Environment Variables](docs/ENVIRONMENT.md) for complete reference.

---

## 🔐 Security Features

### Request Security
- **Size Validation**: Configurable limits with 413 status for oversized requests
- **Timeout Enforcement**: Automatic request timeout handling
- **Rate Limiting**: Per-client rate limiting with configurable thresholds

### Headers & Transport
- **OWASP Headers**: Comprehensive security headers (X-Frame-Options, X-Content-Type-Options, etc.)
- **CSP**: Content Security Policy with strict defaults
- **HSTS**: HTTP Strict Transport Security for production
- **CORS**: Configurable CORS with origin validation

### Data Protection
- **Path Traversal**: Safe path validation for all file operations
- **Log Sanitization**: Automatic redaction of sensitive data (passwords, tokens, keys)
- **API Key Encryption**: Secure client authentication with file-level permissions
- **Request Correlation**: X-Request-ID headers for audit trails

See [Security Fix Summary](docs/SECURITY_FIX_SUMMARY.md) for details.

---

## 📊 Observability

### Structured Logging

All logs are emitted in JSON format when `FLOUDS_LOG_JSON=1`:

<details>
<summary>Example log output</summary>

```json
{
  "timestamp": "2026-01-17T12:34:56.789Z",
  "level": "INFO",
  "logger": "flouds.request",
  "message": "POST /api/v1/embedder/embed -> 200 [45.23ms]",
  "request_id": "abc-123-def-456",
  "tenant_code": "tenant-001",
  "user_id": "user-789",
  "path": "/api/v1/embedder/embed",
  "method": "POST",
  "duration_ms": 45.23,
  "module": "log_context",
  "func": "dispatch",
  "line": 95
}
```

</details>

**Features:**
- Request correlation with X-Request-ID
- Automatic timing and duration tracking
- Endpoint metadata (method, path, status)
- Safe payload sanitization
- Context injection (tenant, user)

### Health Monitoring

The `/api/v1/health` endpoint provides:
- Service status and uptime
- ONNX Runtime availability
- Memory usage and thresholds
- Cache statistics
- Component-level health checks

### Performance Metrics

Built-in monitoring tracks:
- Request latency (p50, p95, p99)
- Cache hit rates
- Memory usage trends
- Model inference times
- Thread pool utilization

---

## 🐳 Docker Deployment

### Build & Helper Script (recommended)

Use the provided script for faster, cached builds with BuildKit:

```powershell
# CPU image (default): uses multi-stage build and layer cache
./build-flouds-ai.ps1 -Tag latest

# GPU image
./build-flouds-ai.ps1 -GPU -Tag latest

# Optional: clean build, multi-arch, and tag latest simultaneously
./build-flouds-ai.ps1 -Tag v1.2.0 -NoCache -Platform linux/amd64 -AlsoTagLatest
```

Key improvements:
- Multi-stage build: dependencies compiled in builder, slim runtime layers only
- BuildKit enabled by default for faster, parallelized builds
- Optional `-NoCache` for clean builds; cache on by default for speed
- Optional `-Platform` for cross-arch (e.g., linux/arm64)
- Optional `-AlsoTagLatest` to tag both versioned and `latest`

### Basic Deployment

```bash
# Build image
docker build -t flouds-ai:latest .

# Run container
docker run -d \
  --name flouds-ai \
  -p 19690:19690 \
  -e FLOUDS_API_ENV=Production \
  -e FLOUDS_ONNX_ROOT=/models \
  -e FLOUDS_LOG_JSON=1 \
  -v /host/models:/models:ro \
  -v /host/logs:/app/logs \
  flouds-ai:latest
```

### Docker Compose

<details>
<summary>Click to expand docker-compose.yml example</summary>

```yaml
version: '3.8'
services:
  flouds-ai:
    build: .
    image: flouds-ai:latest
    container_name: flouds-ai
    ports:
      - "19690:19690"
    environment:
      - FLOUDS_API_ENV=Production
      - FLOUDS_ONNX_ROOT=/models
      - FLOUDS_LOG_JSON=1
      - FLOUDS_LOG_LEVEL=INFO
      - FLOUDS_CACHE_MAX_SIZE=1000
    volumes:
      - ./models:/models:ro
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:19690/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

</details>

### GPU Support

```bash
# Build with GPU support (or use build script: ./build-flouds-ai.ps1 -GPU)
docker build -t flouds-ai:gpu --build-arg GPU=true .

# Run with GPU
docker run -d \
  --gpus all \
  --name flouds-ai-gpu \
  -p 19690:19690 \
  -e FLOUDS_API_ENV=Production \
  -v /host/models:/models:ro \
  flouds-ai:gpu
```

---

## 🚢 Production Deployment

### Pre-Deployment Checklist

- [ ] Set `FLOUDS_API_ENV=Production`
- [ ] Configure `FLOUDS_ONNX_ROOT` with model paths
- [ ] Enable structured logging (`FLOUDS_LOG_JSON=1`)
- [ ] Set appropriate log level (`FLOUDS_LOG_LEVEL=INFO`)
- [ ] Configure rate limiting thresholds
- [ ] Set request size limits
- [ ] Review security headers configuration
- [ ] Configure CORS allowed origins
- [ ] Set up client authentication
- [ ] Verify ONNX Runtime installation
- [ ] Test model loading and inference
- [ ] Configure monitoring/alerting

See [Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md) for complete guide.

### Recommended Production Settings

```bash
# Environment
FLOUDS_API_ENV=Production
FLOUDS_LOG_JSON=1
FLOUDS_LOG_LEVEL=INFO

# Performance
FLOUDS_CACHE_MAX_SIZE=5000
FLOUDS_THREAD_POOL_SIZE=8

# Security
FLOUDS_REQUEST_SIZE_LIMIT=10485760
FLOUDS_RATE_LIMIT_REQUESTS=1000
FLOUDS_RATE_LIMIT_WINDOW=60

# Run with multiple workers
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 19690 \
  --workers 4 \
  --log-config logging.yaml \
  --access-log
```

---

## 📝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone the repository (see Contributing section)
cd Flouds.Py

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Run tests to verify setup
pytest -v

# Expected output: 195 passed
```

> **Note**: Make sure to set `FLOUDS_ONNX_ROOT` and other required environment variables before running tests that require model files.

### Contribution Workflow

1. **Fork and clone the repository**
   ```bash
   # Fork via GitHub UI, then:
   git clone https://github.com/YOUR_USERNAME/Flouds.git
   cd Flouds.Py
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests and linting**
   ```bash
   # Run all tests
   pytest -v --cov=app

   # Run specific test categories
   pytest tests/test_embedder_service.py -v

   # Check code style (if configured)
   flake8 app/ tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub

### Code Guidelines

- **Style**: Follow PEP 8 conventions
- **Testing**: Maintain 100% test pass rate
- **Documentation**: Update docstrings and README for public APIs
- **Security**: Never commit sensitive data (keys, tokens, etc.)
- **Commits**: Write clear, descriptive commit messages

### Areas for Contribution

- 🐛 Bug fixes and issue resolution
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Additional test coverage
- 🚀 Performance optimizations
- 🔒 Security enhancements

See [Contributing Guidelines](CONTRIBUTING.md) for detailed information.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**MIT License Summary:**
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ No liability or warranty

---

## 🤝 Support & Community

### Getting Help

- **Documentation**: Check [docs/](docs/) for comprehensive guides
- **GitHub Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Architecture**: Review [Architecture Overview](docs/ARCHITECTURE.md) for design details
- **Deployment**: See [Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md) for production guidance

### Reporting Issues

When [creating an issue](../../issues/new), please include:
- Flouds AI version
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs (with sensitive data removed)

### Feature Requests

We welcome feature requests via [GitHub Issues](../../issues/new)! Please:
- Check [existing issues](../../issues) first
- Describe the use case and benefit
- Suggest implementation approach if possible
- Consider contributing the feature yourself

---

## 🎯 Roadmap

### Completed
- ✅ Phase A: Security hardening and middleware implementation
- ✅ Phase B: Service refactoring (prompt + embedder modules)
- ✅ Structured JSON logging with request correlation
- ✅ Comprehensive test suite (195 tests, 100% pass rate)
- ✅ Multi-level caching and performance optimizations

### Planned
- 🔄 Additional service refactoring (extractor, base NLP service)
- 🔄 Advanced monitoring dashboards (Prometheus, Grafana)
- 🔄 Distributed tracing integration (OpenTelemetry)
- 🔄 Model versioning and A/B testing support
- 🔄 Kubernetes deployment examples and Helm charts
- 🔄 Multi-model ensemble support

---

<div align="center">

## 🙏 Acknowledgments

**FastAPI** • **ONNX Runtime** • **Transformers Community** • **All Contributors**

---

**Built with ❤️ for production NLP workloads**

[⭐ Star this repo](../../stargazers) • [🐛 Report Issues](../../issues) • [💡 Request Features](../../issues/new)

[Documentation](docs/) • [Contributing](CONTRIBUTING.md) • [License](LICENSE)

---

Made with Python 🐍 • Powered by ONNX ⚡ • Deployed with Docker 🐳

</div>
