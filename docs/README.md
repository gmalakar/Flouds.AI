````markdown
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
    memory_monitor.py            # Memory usage monitoring
    model_cache.py              # LRU model caching
  main.py                       # Enhanced FastAPI application
onnx_loaders/                   # Model export utilities
tests/                          # Comprehensive test suite
.env.example                    # Environment configuration template
```

---

## 📚 Documentation

- **[Embedding Process Flow](EMBEDDING.md)** - Detailed embedding pipeline with pooling, projection, normalization, and quantization
- **[Model Information API](MODEL_INFO_API.md)** - Model availability checks and auto-detected parameters
- **[Environment Variables](ENVIRONMENT.md)** - Complete environment variables reference
- **[Cache Keys Utility](CACHE_KEYS.md)** - Cache key canonicalization and best practices

---
