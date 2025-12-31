> **Note:**  
> This project is under active development and we are looking for more collaborators to help improve and extend Flouds AI!  
> If you're interested in contributing, please reach out or open a pull request.

# Flouds AI

**Flouds AI** is an enterprise-grade Python NLP service framework for text summarization and embedding, built with FastAPI and ONNX runtime. It features comprehensive monitoring, security, and performance optimizations for scalable deployment.

---

## ✨ Key Features

### 🤖 **AI Capabilities**
- **Advanced Text Generation**: Support for seq2seq (T5, BART) and GPT-style models with ONNX optimization
- **Text Summarization**: Seq2seq models with automatic sentence capitalization and advanced sampling
- **High-Performance Embeddings**: Sentence and document embeddings with configurable chunking strategies
- **File Content Extraction**: Extract text from PDF, DOCX, DOC, PPT, Excel, TXT, HTML, CSV files with structured output
- **Model Information API**: Real-time model availability checks, auto-detected parameters, and configuration inspection
- **Auto-Detection**: Automatic detection of dimension, input/output names, and vocab_size from ONNX models
- **Batch Processing**: Async batch operations for high-throughput scenarios
- **Model Optimization**: Optimized ONNX models with automatic fallback and KV caching
- **Advanced Sampling**: Top-k, top-p, and repetition penalty for high-quality text generation

### 🚀 **Enterprise Features**
- **Performance Monitoring**: Real-time system metrics, memory tracking, and performance profiling with optimized rate limiting
- **Advanced Health Checks**: Component-based health monitoring with ONNX, authentication, and memory status
- **Request Validation**: Size limits, timeout handling, and comprehensive error responses
- **Optimized Rate Limiting**: High-performance rate limiting with efficient timestamp counting and batched cleanup
- **Enhanced Security**: CORS protection, log injection prevention, path traversal protection, and encrypted client authentication

### ⚙️ **Configuration & Deployment**
- **Environment-Aware Config**: Development/production configs with environment variable overrides and auto-detection
- **Docker Ready**: Multi-stage builds with GPU support, automated deployment scripts, and optimized images
- **Secure Logging**: Structured logging with rotation, sanitization, and configurable levels
- **Resource Management**: Memory/CPU threshold monitoring with automatic alerts and performance tracking

---

## 📁 Project Structure

```
app/
  config/
    appsettings.json              # Enterprise configuration
    appsettings.development.json  # Development overrides
    onnx_config.json             # ONNX model configurations
    config_loader.py             # Enhanced config management
  middleware/
    rate_limit.py                # Advanced rate limiting
    request_validation.py        # Request size/timeout validation
  models/                        # Pydantic request/response models
  routers/                       # FastAPI route handlers
  services/
    prompt_service.py            # Text generation and processing (renamed from summarizer_service)
    embedder_service.py          # Text embedding and similarity
    extractor_service.py         # File content extraction service
    base_nlp_service.py          # Shared NLP functionality
  utils/
    performance_monitor.py       # System performance tracking
    memory_monitor.py           # Memory usage monitoring
    model_cache.py              # LRU model caching
  main.py                       # Enhanced FastAPI application
onnx_loaders/                   # Model export utilities
tests/                          # Comprehensive test suite
.env.example                    # Environment configuration template
```

---

## ⚙️ Configuration

### Enhanced Configuration System

Flouds AI features a sophisticated configuration system with environment-specific overrides and comprehensive settings.

**Key Configuration Files:**
- `appsettings.json` - Enterprise configuration
- `appsettings.development.json` - Development overrides
- `.env` - Environment variables (copy from `.env.example`)
- `onnx_config.json` - ONNX model configurations (see ONNX Export Guide)

### Core Configuration Sections

```json
{
    "app": {
        "name": "Flouds AI",
        "version": "1.0.0",
        "cors_origins": ["*"],
        "max_request_size": 10485760,
        "request_timeout": 300
    },
    "server": {
        "host": "0.0.0.0",
        "port": 19690,
        "session_provider": "CPUExecutionProvider",
        "keepalive_timeout": 5,
        "graceful_timeout": 30
    },
    "rate_limiting": {
        "enabled": true,
        "requests_per_minute": 200,
        "requests_per_hour": 5000
    },
    "monitoring": {
        "enable_metrics": true,
        "memory_threshold_mb": 1024,
        "cpu_threshold_percent": 80
    },
    "security": {
        "enabled": false,
        "clients_db_path": "clients.db"
    }
}
```

### Environment Variables

**Optional:**
- `FLOUDS_API_ENV` - Environment (Development/Enterprise)
- `APP_DEBUG_MODE` - Enable debug logging (0/1)
- `SERVER_PORT` - Override server port
- `FLOUDS_RATE_LIMIT_PER_MINUTE` - Rate limit override
- `FLOUDS_CORS_ORIGINS` - Comma-separated CORS origins
- `FLOUDS_SECURITY_ENABLED` - Enable client authentication
- `FLOUDS_ENCRYPTION_KEY` - Base64 encoded encryption key for client credentials (if not set, auto-generated and stored in `.encryption_key` file under the folder where the db file resides)

**Note**: `FLOUDS_ONNX_ROOT`, `FLOUDS_ONNX_CONFIG_FILE`, and `FLOUDS_CLIENTS_DB` are automatically set by deployment scripts.

**Development Configuration Example:**
```json
{
    "onnx": {
        "onnx_path": "C:/path/to/your/onnx/models",
        "config_file": "C:/path/to/your/onnx/onnx_config.json"
    }
}
```

---

### onnx_config.json

The `onnx_config.json` file (located in `app/config/onnx_config.json`) is used to configure all ONNX models that you want to use in your application.  
Each entry in this file corresponds to a model you have downloaded and placed in your ONNX model folder.

**Production Optimization (v1.1.0):**
- ✅ **40% size reduction**: Removed parameters that can be auto-detected (293 → 175 lines)
- ✅ **Auto-detection**: `dimension`, `inputnames`, `outputnames`, `vocab_size` detected from ONNX files
- ✅ **Minimal config**: Only required parameters and behavioral overrides needed
- ✅ **Inline docs**: `_comment`, `_auto_detected`, `_defaults` provide configuration guidance

**Configuration Structure:**
- **Key**: The model name (e.g., `"t5-small"`, `"all-MiniLM-L6-v2"`)
- **Value**: A dictionary with model settings:
  - **Required**: `max_length`, `embedder_task`/`summarization_task`, ONNX file paths
  - **Auto-detected**: `dimension`, `inputnames`, `outputnames`, `vocab_size` (optional in config)
  - **Behavioral**: Token IDs, generation parameters, preprocessing flags
  - **Optional**: Optimized model paths, special configurations

**Minimal Example (Auto-Detection):**
```json
"all-MiniLM-L6-v2": {
    "max_length": 256,
    "embedder_task": "fe",
    "normalize": true,
    "pooling_strategy": "mean",
    "chunk_logic": "sentence",
    "encoder_onnx_model": "model.onnx"
}
```

**Full Example (Generation Model):**
```json
"t5-small": {
    "max_length": 512,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "summarization_task": "s2s",
    "legacy_tokenizer": true,
    "encoder_onnx_model": "encoder_model.onnx",
    "decoder_onnx_model": "decoder_model.onnx",
    "special_tokens_map_path": "special_tokens_map.json",
    "num_beams": 4,
    "early_stopping": true
}
```

**Note**: `inputnames`, `outputnames`, and `vocab_size` are now auto-detected from ONNX models. Only include them in config if you need to override the detected values or for models with custom input/output mapping (like BART's `decoder_inputnames`).

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

## 🚀 Quick Start

### Development Setup

1. **Clone and Install Dependencies**
```bash
git clone <repository-url>
cd Flouds.Py
pip install -r app/requirements.txt  # Production only
# OR for development:
pip install -r requirements-dev.txt  # Production + development
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your ONNX model path
```

3. **Export ONNX Models** (see [ONNX Export Guide](#exporting-models-to-onnx))
```bash
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/all-MiniLM-L6-v2" --optimize
```

4. **Run Development Server**
```bash
# Windows
set FLOUDS_API_ENV=Development
set APP_DEBUG_MODE=1
set FLOUDS_ONNX_ROOT=path/to/your/onnx
python -m app.main

# Linux/macOS
export FLOUDS_API_ENV=Development
export APP_DEBUG_MODE=1
export FLOUDS_ONNX_ROOT=/path/to/your/onnx
python -m app.main
```

### Enterprise Deployment

```bash
# Using Docker (recommended)
docker run -p 19690:19690 \
  -v /path/to/onnx:/flouds-ai/onnx \
  -e FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
  gmalakar/flouds-ai-cpu:latest

# Direct deployment
export FLOUDS_API_ENV=Enterprise
export FLOUDS_ONNX_ROOT=/path/to/onnx
python -m app.main
```

---

## Exporting Models to ONNX

Flouds AI provides two export methods for converting HuggingFace models to ONNX format:

### Method 1: Standard Export (`export_model.py`)
Uses optimum's built-in export with automatic fallbacks and optimizations.

```bash
# Embedding models
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/all-MiniLM-L6-v2" --optimize
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/all-mpnet-base-v2" --optimize

# Summarization models  
python onnx_loaders/export_model.py --model_for "s2s" --model_name "t5-small" --optimize --task "seq2seq-lm"
python onnx_loaders/export_model.py --model_for "s2s" --model_name "facebook/bart-large-cnn" --optimize --task "text2text-generation-with-past" --use_cache
```

### Method 2: Manual Export (`export_model_v2.py`) ⭐ **Recommended**
Uses direct optimum main_export with enhanced error handling and validation.

```bash
# Embedding models
python onnx_loaders/export_model_v2.py --model_for "fe" --model_name "sentence-transformers/all-MiniLM-L6-v2" --task "feature-extraction" --optimize
python onnx_loaders/export_model_v2.py --model_for "fe" --model_name "sentence-transformers/all-mpnet-base-v2" --task "feature-extraction" --opset_version 17

# Summarization models
python onnx_loaders/export_model_v2.py --model_for "s2s" --model_name "t5-small" --task "seq2seq-lm" --optimize
python onnx_loaders/export_model_v2.py --model_for "s2s" --model_name "facebook/bart-large-cnn" --task "text2text-generation-with-past" --optimize
```

### Export Parameters
- `"fe"` = feature extraction (embedding)
- `"s2s"` = sequence-to-sequence (summarization)  
- `"sc"` = sequence classification
- `--task` = **Required for v2** - HuggingFace task type
- `--optimize` = Enable ONNX graph optimizations
- `--opset_version` = ONNX opset version (default: 14)
- `--model_folder` = Custom output folder name

### Key Improvements in v2
- ✅ **Enhanced Error Handling**: Detailed stack traces and corruption detection
- ✅ **Model Validation**: Automatic verification and cleanup of corrupted files
- ✅ **Flexible ONNX Opset**: Configurable opset versions (11-17)
- ✅ **Better Optimization**: Graceful handling of unsupported model types
- ✅ **Mandatory Task Parameter**: Prevents export configuration errors

---

## 🔐 Authentication

### Authentication Methods

Flouds AI supports two authentication methods:

#### 1. Authorization Header (Production & Development)
```bash
curl -H "Authorization: Bearer <client_id>|<client_secret>" \
  http://localhost:19690/api/v1/summarizer/summarize
```

#### 2. Query Parameter (Development Only)
```bash
curl "http://localhost:19690/api/v1/summarizer/summarize?token=<client_id>|<client_secret>"
```

**Note**: Query parameter authentication is only available in development mode for convenience. Production deployments require the Authorization header for security.

### Client-Based Authentication

Flouds AI uses a modern client-based authentication system with encrypted storage.

**Token Format**: `client_id|client_secret`

#### Auto-Generated Admin
On first startup, an admin client is automatically created:
- Credentials logged to console and saved to `admin_credentials.txt`
- Use admin token to manage other clients

#### Client Management CLI

```bash
# Add new API client
python generate_token.py add my-app --type api_user

# Add admin client
python generate_token.py add admin-user --type admin

# List all clients
python generate_token.py list

# Remove client
python generate_token.py remove my-app
```

#### Admin API Endpoints

```bash
# Generate client key (admin only)
curl -H "Authorization: Bearer <admin_client_id>|<admin_client_secret>" \
  -X POST "http://localhost:19690/api/v1/admin/generate-key" \
  -d '{"client_id": "<new_client_id>"}'

# List clients (admin only)
curl -H "Authorization: Bearer <admin_client_id>|<admin_client_secret>" \
  "http://localhost:19690/api/v1/admin/clients"

# Remove client (admin only)
curl -H "Authorization: Bearer <admin_client_id>|<admin_client_secret>" \
  -X DELETE "http://localhost:19690/api/v1/admin/remove-client/<client_id>"
```

---

## 📚 API Usage

### REST API Endpoints

Flouds AI provides a comprehensive REST API with automatic documentation at `/docs`.

#### Model Information & Discovery

```bash
# Get detailed model information
curl -X GET "http://localhost:19690/api/v1/model/info?model=all-MiniLM-L6-v2" \
  -H "Authorization: Bearer <client_id>|<client_secret>"

# List all available models
curl -X GET "http://localhost:19690/api/v1/models/list" \
  -H "Authorization: Bearer <client_id>|<client_secret>"
```

**Model Information Response:**
```json
{
  "success": true,
  "model": "all-MiniLM-L6-v2",
  "model_type": "embedding",
  "model_available": true,
  "onnx_file_available": true,
  "auto_detected_params": {
    "dimension": 384,
    "inputnames": ["input_ids", "attention_mask"],
    "outputnames": ["token_embeddings", "sentence_embedding"]
  },
  "config_params": {
    "max_length": 256,
    "pooling_strategy": "mean",
    "normalize": true
  },
  "required_params": ["max_length", "embedder_task", "normalize"],
  "default_params": {"force_pooling": false, "chunk_overlap": 0}
}
```

**Key Features:**
- ✅ **Auto-Detection**: Dimension, input/output names, vocab_size detected from ONNX models
- ✅ **Type-Specific Filtering**: Only relevant parameters returned for each model type
- ✅ **Security**: Internal file paths not exposed
- ✅ **Real-Time Status**: Check model availability before inference

See [MODEL_INFO_API.md](MODEL_INFO_API.md) for complete documentation.

#### Text Generation/Summarization

```bash
# Single text generation
curl -X POST "http://localhost:19690/api/v1/summarizer/summarize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_id>|<client_secret>" \
  -d '{
    "model": "t5-small",
    "input": "Your text to process here...",
    "temperature": 0.7
  }'

# Development mode: Query parameter authentication (less secure)
curl -X POST "http://localhost:19690/api/v1/summarizer/summarize?token=<client_id>|<client_secret>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "t5-small",
    "input": "Your text to process here...",
    "temperature": 0.7
  }'

# Batch text generation
curl -X POST "http://localhost:19690/api/v1/summarizer/summarize_batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_id>|<client_secret>" \
  -d '{
    "model": "t5-small",
    "inputs": ["Text 1...", "Text 2..."],
    "temperature": 0.5
  }'

# GPT-style text generation
curl -X POST "http://localhost:19690/api/v1/sendprompt" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_id>|<client_secret>" \
  -d '{
    "model": "distilgpt2",
    "input": "Once upon a time",
    "temperature": 0.8
  }'
```

#### File Content Extraction

```bash
# Extract text from base64 encoded file
curl -X POST "http://localhost:19690/api/v1/extract/extract" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_id>|<client_secret>" \
  -d '{
    "file_content": "<base64_encoded_file>",
    "extention": "pdf"
  }'

# Extract text from multipart file upload
curl -X POST "http://localhost:19690/api/v1/extract/extract_file" \
  -H "Authorization: Bearer <client_id>|<client_secret>" \
  -F "file=@document.pdf" \
  -F "extension=pdf"
```

**Supported File Types:**
- **PDF**: Extracts text from each page with page numbers
- **DOCX/DOC**: Extracts text from each paragraph with paragraph numbers
- **PowerPoint (PPT/PPTX)**: Extracts text from each slide with slide numbers
- **Excel (XLS/XLSX)**: Extracts data from each sheet with sheet numbers
- **TXT/MD**: Plain text extraction
- **HTML/HTM**: Extracts clean text content
- **CSV**: Extracts each row as structured data

**Response Format:**
```json
{
  "success": true,
  "message": "Text extracted successfully",
  "results": [
    {
      "content": "Extracted text content",
      "item_number": 1,
      "content_as": "pages"
    }
  ],
  "time_taken": 0.25
}
```

```bash
# Single embedding
curl -X POST "http://localhost:19690/api/v1/embedder/embed" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_id>|<client_secret>" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "input": "Text to embed",
    "projected_dimension": 128,
    "pooling_strategy": "mean"
  }'

# Batch embedding with custom parameters
curl -X POST "http://localhost:19690/api/v1/embedder/embed_batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_id>|<client_secret>" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "inputs": ["Text 1", "Text 2"],
    "projected_dimension": 64,
    "pooling_strategy": "cls",
    "normalize": true,
    "max_length": 256
  }'
```

### Python SDK Usage

```python
# Text Generation/Summarization
from app.models.prompt_request import PromptRequest
from app.services.prompt_service import PromptProcessor

request = PromptRequest(
    model="t5-small",
    input="Your text to summarize",
    temperature=0.7
)
response = PromptProcessor.process_prompt(request)
print(response.results[0])

# Embedding
from app.models.embedding_request import EmbeddingRequest
from app.services.embedder_service import SentenceTransformer

request = EmbeddingRequest(
    model="all-MiniLM-L6-v2",
    input="Text to embed",
    projected_dimension=128
)
response = SentenceTransformer.embed_text(
    text=request.input,
    model_to_use=request.model,
    projected_dimension=request.projected_dimension
)
print(response.results)
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Pull latest image
docker pull gmalakar/flouds-ai-cpu:latest

# Run with your ONNX models
docker run -d \
  --name flouds-ai \
  -p 19690:19690 \
  -v /path/to/your/onnx:/flouds-ai/onnx \
  -e FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
  -e FLOUDS_API_ENV=Enterprise \
  --restart unless-stopped \
  gmalakar/flouds-ai-cpu:latest
```

### Build Custom Image

```bash
# CPU version
docker build -t flouds-ai-cpu .

# GPU version (requires NVIDIA Docker)
docker build --build-arg GPU=true -t flouds-ai-gpu .
```

### Enterprise Docker Compose

```yaml
version: '3.8'
services:
  flouds-ai:
    image: gmalakar/flouds-ai-cpu:latest
    ports:
      - "19690:19690"
    volumes:
      - ./onnx:/flouds-ai/onnx
      - ./logs:/flouds-ai/logs
    environment:
      - FLOUDS_ONNX_ROOT=/flouds-ai/onnx
      - FLOUDS_API_ENV=Enterprise
      - FLOUDS_RATE_LIMIT_PER_MINUTE=500
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:19690/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### Automated Deployment Scripts

Flouds AI includes comprehensive deployment scripts for easy Docker container management.

#### PowerShell Script (Windows) - `start-flouds-ai.ps1`

**Basic Usage:**
```powershell
# Start with default settings
.\start-flouds-ai.ps1

# Start with custom environment file
.\start-flouds-ai.ps1 -EnvFile .env.production

# Force restart existing container
.\start-flouds-ai.ps1 -Force

# Build image locally and start
.\start-flouds-ai.ps1 -BuildImage -Force

# Use GPU image
.\start-flouds-ai.ps1 -GPU

# Always pull latest image
.\start-flouds-ai.ps1 -PullAlways
```

**Parameters:**
- `-EnvFile` - Path to environment file (default: `.env`)
- `-InstanceName` - Container name (default: `flouds-ai-instance`)
- `-ImageName` - Docker image name (default: `gmalakar/flouds-ai-cpu`)
- `-Tag` - Image tag (default: `latest`)
- `-GPU` - Use GPU image instead of CPU
- `-Force` - Force restart if container exists
- `-BuildImage` - Build image locally before starting
- `-PullAlways` - Always pull latest image from registry

#### Bash Script (Linux/macOS) - `start-flouds-ai.sh`

**Basic Usage:**
```bash
# Start with default settings
./start-flouds-ai.sh

# Start with custom environment file
./start-flouds-ai.sh --env-file .env.production

# Force restart existing container
./start-flouds-ai.sh --force

# Build image locally and start
./start-flouds-ai.sh --build --force

# Development mode with GPU
./start-flouds-ai.sh --gpu --development
```

**Parameters:**
- `--env-file` - Path to environment file (default: `.env`)
- `--instance` - Container name (default: `flouds-ai-instance`)
- `--image` - Docker image name (default: `gmalakar/flouds-ai-cpu`)
- `--tag` - Image tag (default: `latest`)
- `--gpu` - Use GPU image instead of CPU
- `--force` - Force restart if container exists
- `--build` - Build image locally before starting
- `--pull-always` - Always pull latest image
- `--development` - Run in development mode

#### Build Script - `build-flouds-ai.ps1`

**Usage:**
```powershell
# Build CPU image
.\build-flouds-ai.ps1

# Build GPU image with custom tag
.\build-flouds-ai.ps1 -Tag v1.2.0 -GPU

# Build and push to registry
.\build-flouds-ai.ps1 -PushImage

# Force rebuild existing image
.\build-flouds-ai.ps1 -Force
```

**Parameters:**
- `-Tag` - Image tag (default: `latest`)
- `-GPU` - Build with GPU support
- `-PushImage` - Push to Docker registry after build
- `-Force` - Force rebuild even if image exists

#### Required Environment File Structure

Create a `.env` file with the following required variables:

```bash
# Required: ONNX model configuration
FLOUDS_ONNX_CONFIG_FILE_AT_HOST=/path/to/your/onnx_config.json
FLOUDS_ONNX_MODEL_PATH_AT_HOST=/path/to/your/onnx/models

# Optional: Log persistence
FLOUDS_LOG_PATH_AT_HOST=/path/to/logs

# Optional: Docker platform
DOCKER_PLATFORM=linux/amd64
```

#### Script Features

**Automatic Setup:**
- ✅ Docker availability checking
- ✅ Environment file validation
- ✅ Directory permission management
- ✅ Network creation and management
- ✅ Container lifecycle management

**Volume Mapping:**
- ✅ ONNX models directory (read-only)
- ✅ Configuration files (read-only)
- ✅ Log directory (read-write)
- ✅ Automatic directory creation

**Error Handling:**
- ✅ Comprehensive validation
- ✅ Graceful error messages
- ✅ Automatic cleanup on failure
- ✅ Interactive confirmations

#### Example Complete Workflow

```powershell
# 1. Setup environment
cp .env.example .env
# Edit .env with your paths

# 2. Build custom image (optional)
.\build-flouds-ai.ps1 -Tag production

# 3. Start container
.\start-flouds-ai.ps1 -Tag production -Force

# 4. View logs
docker logs -f flouds-ai-instance

# 5. Test API
curl http://localhost:19690/api/v1/health
```

---

## 📊 Monitoring & Health Checks

### Component-Based Health System

Flouds AI features a comprehensive health monitoring system that checks individual components:

```bash
# Comprehensive health check with component status
curl http://localhost:19690/api/v1/health

# Detailed system information with metrics
curl http://localhost:19690/api/v1/health/detailed

# Performance metrics for rate limiting and authentication
curl http://localhost:19690/api/v1/health/performance

# Kubernetes probes
curl http://localhost:19690/api/v1/health/ready   # Readiness probe
curl http://localhost:19690/api/v1/health/live    # Liveness probe
```

### Health Response Format

```json
{
  "status": "healthy",
  "service": "Flouds AI",
  "version": "1.0.0",
  "timestamp": "2025-01-27T...",
  "uptime_seconds": 3600,
  "components": {
    "onnx": "healthy",
    "authentication": "healthy",
    "memory": "healthy"
  }
}
```

### Performance Monitoring

Flouds AI includes comprehensive monitoring with significant performance optimizations:

- **Optimized Rate Limiting**: 60% faster through efficient timestamp counting and batched operations
- **Enhanced Authentication**: 40% faster through LRU caching and set-based token lookups
- **Real-time Metrics**: Memory usage, CPU utilization, request timing with performance tracking
- **Resource Thresholds**: Configurable alerts for memory/CPU limits with component-based health checks
- **Request Analytics**: Slow request detection and logging with sanitized outputs
- **Model Cache Monitoring**: Track cached models and sessions with sliding expiration

### Logging

```bash
# View logs using deployment scripts
# PowerShell
.\start-flouds-ai.ps1  # Will prompt to view logs after startup

# Bash
./start-flouds-ai.sh   # Will prompt to view logs after startup

# Manual log viewing
docker logs -f flouds-ai-instance

# Log files (direct deployment)
tail -f logs/flouds-ai-$(date +%Y-%m-%d).log
```

### Container Management

```bash
# Using deployment scripts (recommended)
.\start-flouds-ai.ps1 -Force  # Restart container
./start-flouds-ai.sh --force   # Restart container

# Manual Docker commands
docker stop flouds-ai-instance
docker start flouds-ai-instance
docker restart flouds-ai-instance
docker rm flouds-ai-instance
```

---

## 🛡️ Security Features

### Enhanced Security Implementation

Flouds AI implements comprehensive security measures:

- **Log Injection Prevention**: All user inputs sanitized before logging using `sanitize_for_log()`
- **Path Traversal Protection**: Safe path validation for all file operations
- **Encrypted Authentication**: Client credentials encrypted with Fernet encryption
- **Rate Limiting Protection**: Optimized rate limiting prevents abuse and DoS attacks
- **Input Validation**: Request size limits, timeout handling, and comprehensive sanitization
- **CORS Protection**: Configurable CORS origins with trusted host validation

### Security Best Practices

```python
# All logging uses sanitization
logger.info("Processing request for client: %s", sanitize_for_log(client_id))

# Path operations use validation
safe_path = validate_safe_path(user_path, base_directory)

# Authentication uses encrypted storage
encrypted_secret = fernet.encrypt(client_secret.encode())
```

---

## 🔧 Advanced Configuration

### Complete Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FLOUDS_ONNX_ROOT` | ONNX models directory | - | ✅ |
| `FLOUDS_API_ENV` | Environment (Development/Enterprise) | Enterprise | ❌ |
| `APP_DEBUG_MODE` | Enable debug logging (0/1) | 0 | ❌ |
| `SERVER_PORT` | Server port | 19690 | ❌ |
| `SERVER_HOST` | Server host | 0.0.0.0 | ❌ |
| `FLOUDS_RATE_LIMIT_PER_MINUTE` | Rate limit per minute | 200 | ❌ |
| `FLOUDS_RATE_LIMIT_PER_HOUR` | Rate limit per hour | 5000 | ❌ |
| `FLOUDS_MAX_REQUEST_SIZE` | Max request size (bytes) | 10485760 | ❌ |
| `FLOUDS_MODEL_SESSION_PROVIDER` | ONNX provider | CPUExecutionProvider | ❌ |
| `FLOUDS_CORS_ORIGINS` | CORS origins (comma-separated) | * | ❌ |
| `FLOUDS_ENCRYPTION_KEY` | Base64 encoded encryption key | Auto-generated file | ❌ |

### Performance Tuning

```bash
# High-performance configuration
export FLOUDS_MODEL_CACHE_SIZE=10
export FLOUDS_MODEL_CACHE_TTL=7200
export FLOUDS_RATE_LIMIT_PER_MINUTE=1000
export FLOUDS_MAX_REQUEST_SIZE=52428800  # 50MB

# GPU acceleration
export FLOUDS_MODEL_SESSION_PROVIDER=CUDAExecutionProvider
```

---

## 🎯 Model Management

### ONNX Model Optimization

Flouds AI provides advanced model optimization features:

- **Automatic Optimization**: Graph-optimized models with 20-50% performance improvement
- **KV Caching**: Faster seq2seq inference with decoder caching
- **Model Fallback**: Automatic fallback to regular models if optimized versions fail
- **Legacy Support**: Handles tokenizer compatibility across transformers versions

### Model Configuration

```json
{
  "t5-small": {
    "use_optimized": true,
    "legacy_tokenizer": false,
    "use_seq2seqlm": true,
    "dimension": 512,
    "max_length": 512
  }
}
```

### Performance Features

- **Optimized Model Cache**: Intelligent LRU caching with sliding expiration and automatic cleanup
- **Enhanced Rate Limiting**: Efficient timestamp counting with 60% performance improvement
- **Optimized Authentication**: LRU token parsing cache with set-based lookups (40% faster)
- **Thread-Safe Operations**: Concurrent request handling with performance tracking
- **Memory Management**: Automatic cleanup and resource monitoring with component health checks
- **Batch Optimization**: Efficient batch processing for high throughput scenarios

---

## 🔌 API Reference

### API Versioning

Flouds AI uses API versioning with the `/api/v1` prefix for all endpoints:

- **Current Version**: v1
- **Base URL**: `http://localhost:19690/api/v1`
- **Documentation**: `/api/v1/docs`
- **OpenAPI Spec**: `/api/v1/openapi.json`

### Core Endpoints

| Endpoint | Method | Description | Rate Limited |
|----------|--------|-------------|-------------|
| `/api/v1/summarizer/summarize` | POST | Single text generation/summarization | ✅ |
| `/api/v1/summarizer/summarize_batch` | POST | Batch text generation/summarization | ✅ |
| `/api/v1/sendprompt` | POST | GPT-style text generation | ✅ |
| `/api/v1/embedder/embed` | POST | Single text embedding | ✅ |
| `/api/v1/embedder/embed_batch` | POST | Batch text embedding | ✅ |
| `/api/v1/extract/extract` | POST | Extract text from base64 file | ✅ |
| `/api/v1/extract/extract_file` | POST | Extract text from multipart upload | ✅ |
| `/api/v1/health` | GET | Basic health check | ❌ |
| `/api/v1/health/detailed` | GET | Detailed system info | ❌ |
| `/api/v1/health/ready` | GET | Readiness probe | ❌ |
| `/api/v1/health/live` | GET | Liveness probe | ❌ |
| `/api/v1/docs` | GET | Interactive API docs | ❌ |
| `/api/v1/health/performance` | GET | Performance metrics | ❌ |

### Parameter Override Behavior

Flouds AI uses a **request-first, config-fallback** approach:

1. **Request parameter provided**: Uses the request parameter value
2. **Request parameter is `null`**: Falls back to model configuration value
3. **Both are `null`**: Uses system defaults

**Available Override Parameters:**
- `projected_dimension`: Target embedding dimension (e.g., 64, 128, 256)
- `pooling_strategy`: Pooling method (`"mean"`, `"max"`, `"cls"`, `"first"`, `"last"`, `"none"`)
- `max_length`: Maximum token length for input text
- `chunk_logic`: Text chunking strategy (`"sentence"`, `"paragraph"`, `"fixed"`)
- `chunk_overlap`: Number of overlapping tokens/sentences between chunks
- `chunk_size`: Fixed chunk size in tokens (for `"fixed"` chunking)
- `normalize`: Normalize embedding vectors to unit length
- `force_pooling`: Force pooling even for single-token sequences
- `legacy_tokenizer`: Use legacy tokenizer for older models
- `lowercase`: Convert input text to lowercase
- `remove_emojis`: Remove emojis and non-ASCII characters
- `use_optimized`: Use optimized ONNX model if available

### Response Headers

- `X-Processing-Time`: Request processing time in seconds
- `X-RateLimit-Remaining-Minute`: Remaining requests this minute
- `X-RateLimit-Remaining-Hour`: Remaining requests this hour

### Response Format

All embedding responses include a `used_parameters` field showing the actual parameter values used:

```json
{
  "success": true,
  "message": "Embedding generated successfully",
  "model": "sentence-t5-base",
  "results": [...],
  "used_parameters": {
    "pooling_strategy": "cls",
    "projected_dimension": 64,
    "max_length": 256,
    "normalize": true,
    "force_pooling": true
  },
  "time_taken": 0.45
}
```

---

## 🧪 Testing & Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_embedder_service.py -v
```

### Code Quality

```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

### Development Tools

- **Pre-commit hooks**: Automatic code formatting and linting
- **Comprehensive test suite**: Unit tests with mocking
- **Performance profiling**: Built-in performance monitoring
- **Hot reload**: Development server with auto-reload

---

## 🤝 Contributing

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/Flouds.Py.git
   cd Flouds.Py
   ```

2. **Setup Development Environment**
   ```bash
   pip install -r requirements-dev.txt  # Includes production + dev dependencies
   pre-commit install  # Install pre-commit hooks
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-amazing-feature
   ```

4. **Make Changes & Test**
   ```bash
   pytest  # Run tests
   black app/ tests/  # Format code
   ```

5. **Submit Pull Request**
   - Ensure all tests pass
   - Add tests for new features
   - Update documentation
   - Follow conventional commit messages

---

## 📈 Performance Benchmarks

### Optimized Performance (After Improvements)

| Operation | Model | Throughput | Latency | Improvement |
|-----------|-------|------------|---------|-------------|
| Summarization | t5-small | 80 req/min | 0.9s | +60% faster |
| Embedding | all-MiniLM-L6-v2 | 280 req/min | 0.2s | +40% faster |
| Batch Summarization (10x) | t5-small | 200 req/min | 3.2s | +33% faster |
| Batch Embedding (10x) | all-MiniLM-L6-v2 | 650 req/min | 0.9s | +30% faster |
| Rate Limiting Check | - | - | 0.1ms | +60% faster |
| Authentication | - | - | 0.05ms | +40% faster |

*Benchmarks on Intel i7-10700K, 32GB RAM, CPU-only*

### Performance Optimizations Implemented

- **Rate Limiting**: Optimized timestamp counting and batched cleanup operations
- **Authentication**: LRU token parsing cache and set-based token lookups
- **Model Operations**: Improved caching strategies and resource management
- **Memory Management**: Efficient cleanup and sliding expiration mechanisms

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model ecosystem
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Uvicorn](https://www.uvicorn.org/) - ASGI server

---

## 👨‍💻 Maintainer

**Goutam Malakar**  
📧 Contact: [Create an issue](https://github.com/your-repo/issues) for support

---

## 🔍 Code Quality & Security

### Recent Improvements

- ✅ **Enhanced ONNX Export** - New export_model_v2.py with improved error handling and validation
- ✅ **Smart Cache Management** - Only clear cache when memory is low AND item not cached
- ✅ **Model Validation** - Automatic verification and cleanup of corrupted ONNX files
- ✅ **Flexible Opset Support** - Configurable ONNX opset versions (11-17) for compatibility
- ✅ **100% Log Sanitization Coverage** - All user inputs sanitized to prevent log injection
- ✅ **Path Traversal Protection** - Safe path validation for all file operations
- ✅ **Performance Optimizations** - 60% faster rate limiting, 40% faster authentication
- ✅ **Component-Based Health Checks** - Individual monitoring of ONNX, auth, and memory
- ✅ **Enhanced Error Handling** - Specific exception types for better debugging
- ✅ **Encrypted Client Storage** - Secure credential management with Fernet encryption
- ✅ **Optimized Caching** - LRU caches with sliding expiration and automatic cleanup
- ✅ **Resource Leak Prevention** - Proper cleanup and resource management
- ✅ **Service Refactoring** - Renamed summarizer_service to prompt_service for clarity
- ✅ **Memory Optimizations** - Improved array operations and reduced object creation
- ✅ **Dead Code Removal** - Cleaned up unused variables and model classes
- ✅ **Enhanced File Extraction** - Support for DOC, PPT/PPTX, XLS/XLSX formats with structured output
- ✅ **File Content Extraction** - New ExtractorService for PDF, DOCX, TXT, HTML, CSV processing
- ✅ **Multipart Upload Support** - Both base64 and multipart file upload endpoints
- ✅ **Structured Extraction Output** - Item-based extraction with content type identification

### Security Compliance

- **CWE-117**: Log injection prevention through input sanitization
- **CWE-22**: Path traversal protection with safe path validation
- **CWE-79**: XSS prevention through proper input handling
- **Authentication**: Encrypted storage and secure token management
- **Rate Limiting**: DoS protection with optimized performance

---

*Built with ❤️ and 🛡️ security for the AI community*