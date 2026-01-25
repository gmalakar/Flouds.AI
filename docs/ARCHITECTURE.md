# Flouds.Py Architecture Guide

**Last Updated**: January 17, 2026
**Version**: 1.0

## Table of Contents
1. [System Overview](#system-overview)
2. [Layer Architecture](#layer-architecture)
3. [Component Responsibilities](#component-responsibilities)
4. [Data Flow](#data-flow)
5. [Security Model](#security-model)
6. [Performance Characteristics](#performance-characteristics)
7. [Deployment Architecture](#deployment-architecture)

---

## System Overview

Flouds.Py is a **FastAPI-based AI service** that provides:
- 📄 **Text Summarization** using ONNX-optimized models
- 🔤 **Text Embedding** with multiple pooling strategies
- 📦 **Document Extraction** from various file formats
- 🔐 **Multi-tenant** architecture with strong data isolation
- ⚡ **High-performance** inference with caching and pooling

### Technology Stack
```
┌─────────────────────────────────────────────┐
│        FastAPI + Uvicorn                    │
│        (Async HTTP server)                  │
├─────────────────────────────────────────────┤
│        Python 3.11+                         │
├─────────────────────────────────────────────┤
│   ┌──────────────┬──────────────┐           │
│   │   ONNX       │ Transformers │           │
│   │   Runtime    │ + Optimum    │           │
│   └──────────────┴──────────────┘           │
├─────────────────────────────────────────────┤
│   ┌──────────────┬──────────────┐           │
│   │   SQLite     │ In-Memory    │           │
│   │   Config DB  │ Caches       │           │
│   └──────────────┴──────────────┘           │
└─────────────────────────────────────────────┘
```

---

## Layer Architecture

### **5-Layer Hexagonal Architecture**

```
┌───────────────────────────────────────────────────────────┐
│                    HTTP Clients                           │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│                    Middleware Layer                       │
│  ┌──────────┬──────────┬──────────────┬────────────────┐  │
│  │Auth      │Rate Limit│Security      │Request Size   │  │
│  │Tenant    │Headers   │Validation    │Limit          │  │
│  └──────────┴──────────┴──────────────┴────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Router Layer (HTTP)                    │
│  ┌──────────────┬──────────────┬──────────────────────┐  │
│  │Summarizer    │Embedder      │Extractor            │  │
│  │Endpoints     │Endpoints     │Endpoints            │  │
│  └──────────────┴──────────────┴──────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                          │
│  ┌──────────────┬──────────────┬──────────────────────┐  │
│  │PromptService│EmbedderService│ExtractorService    │  │
│  │(1716 lines) │(1514 lines)  │                     │  │
│  └──────────────┴──────────────┴──────────────────────┘  │
│                   ▲                                       │
│         ┌─────────┴──────────┐                           │
│         │ BaseNLPService     │                           │
│         │ (1144 lines)       │                           │
│         │ - ONNX sessions    │                           │
│         │ - Tokenization     │                           │
│         │ - Model loading    │                           │
│         └────────────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│                    Data Access Layer                      │
│  ┌──────────────┬──────────────┬──────────────────────┐  │
│  │Config Service│Cache Manager │Model Manager        │  │
│  │(622 lines)   │(273 lines)   │                     │  │
│  └──────────────┴──────────────┴──────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    External Resources                    │
│  ┌──────────────┬──────────────┬──────────────────────┐  │
│  │SQLite DB     │ONNX Models   │File System           │  │
│  │Encryption    │ (.onnx files)│(Documents)           │  │
│  └──────────────┴──────────────┴──────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### **Layer Responsibilities**

#### 1. **Middleware Layer** - Request Processing
**Files**: `middleware/auth.py`, `middleware/rate_limit.py`, `middleware/tenant_security.py`

**Responsibilities**:
- ✅ Authentication: Bearer token validation
- ✅ Rate limiting: Per-IP request throttling
- ✅ Tenant isolation: Verify tenant codes
- ✅ Request validation: Size limits, headers
- ✅ Security: CORS, headers, path validation

**Key Classes**:
- `AuthMiddleware`: JWT/token validation
- `RateLimitMiddleware`: In-memory request tracking (O(log n) with bisect)
- `TenantTrustedHostMiddleware`: Tenant-scoped CORS

---

#### 2. **Router Layer** - HTTP Endpoints
**Files**: `routers/*.py` (11 files)

**Responsibilities**:
- ✅ HTTP request handling
- ✅ Request/response serialization
- ✅ Dependency injection
- ✅ OpenAPI documentation

**Key Routers**:
- `routers/summarizer.py` - Text summarization endpoints
- `routers/embedder.py` - Text embedding endpoints
- `routers/extractor.py` - Document extraction endpoints
- `routers/model_info.py` - Model metadata endpoints

**Endpoint Pattern**:
```python
@router.post("/embed")
async def embed_text(request: EmbedRequest, token: str = Depends(get_token)) -> EmbedResponse:
    # 1. Validate input
    # 2. Call service
    # 3. Return response
```

---

#### 3. **Service Layer** - Business Logic
**Files**: `services/*.py` (8 files)

**Responsibilities**:
- ✅ ONNX model inference
- ✅ Batch processing
- ✅ Caching strategies
- ✅ Error handling
- ✅ Performance optimization

**Key Services**:

| Service | Responsibility | Lines | Tech |
|---------|-----------------|-------|------|
| `PromptService` | Text summarization | 1,716 | ONNX, Transformers |
| `EmbedderService` | Sentence embeddings | 1,514 | ONNX, Pooling |
| `ExtractorService` | Document extraction | 400+ | PyPDF, etc. |
| `BaseNLPService` | Common NLP operations | 1,144 | ONNX sessions |
| `ConfigService` | Configuration management | 622 | SQLite, Encryption |
| `KeyManager` | Encryption keys | 644 | Fernet |

---

#### 4. **Data Access Layer** - Caching & Storage
**Files**: `utils/cache_manager.py`, `services/config_service.py`

**Responsibilities**:
- ✅ Cache lifecycle management
- ✅ Memory monitoring
- ✅ Configuration persistence
- ✅ Encryption at rest

**Caching Strategy** (3-level):
```
L1: In-Memory        (ConcurrentDict)      - Fast, per-instance
L2: Thread-Local     (ThreadLocal cache)   - Thread-safe ONNX sessions
L3: Disk             (ONNX model files)    - Persistent, large
```

**Cache Management**:
- Throttled memory checks (5-second interval)
- Automatic cleanup when memory < 1GB
- Per-tenant cache isolation

---

#### 5. **External Resources** - Persistence
**Components**:
- 📁 **SQLite Database**: Configuration, tenant settings
- 📦 **ONNX Models**: Model artifacts (.onnx files)
- 📄 **File System**: Document uploads, logs
- 🔐 **Encryption**: Fernet key-based encryption

---

## Component Responsibilities

### Core Classes

#### `PromptProcessor` (1,716 lines) ⚠️
**Inherits from**: `BaseNLPService`

**Responsibilities**:
- Model loading and lifecycle
- Tokenization and encoding
- ONNX session management
- Batch processing
- Decoding strategies
- Response formatting

**Issues**:
- ❌ Too large (God class)
- ❌ Mixed concerns
- ❌ Difficult to test

**Future Refactoring**: Split into 7 modules

---

#### `SentenceTransformer` (1,514 lines) ⚠️
**Inherits from**: `BaseNLPService`

**Responsibilities**:
- Sentence embedding generation
- Text preprocessing (Unicode normalization)
- Vector merging (mean/max pooling)
- Chunking strategies
- Batch limiting

**Issues**:
- ❌ Removed asyncio (performance concern)
- ❌ Synchronous batch processing
- ❌ Code duplication with extractor

**Future Refactoring**: Enable async/await, split concerns

---

#### `BaseNLPService` (1,144 lines)
**Base class for all NLP services**

**Provides**:
- ONNX session management
- Tokenizer initialization
- Model loading
- Cache management
- Path validation

---

#### `ConfigService` (622 lines)
**Manages application configuration**

**Features**:
- SQLite-backed configuration store
- Tenant-scoped settings
- Encryption at rest
- In-memory caching
- Thread-safe access

---

#### `CacheManager` (273 lines)
**Centralized cache lifecycle management**

**Features**:
- Memory-aware cache cleanup
- Throttled memory checks
- Cache metrics
- Multi-tier invalidation
- Thread-safe operations

---

### Utility Classes

#### `ConcurrentDict` (6,845 bytes)
Thread-safe dictionary for cache storage

#### `RateLimitMiddleware` (217 lines)
Per-IP rate limiting with O(log n) lookups

#### `EncryptionManager` (via config_service)
Fernet-based encryption for sensitive data

---

## Data Flow

### Text Embedding Flow

```
HTTP Request
    │
    ▼
┌─────────────────────────┐
│ Router: /embed          │
│ - Validate request      │
│ - Extract auth token    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Middleware Chain        │
│ - Auth verification     │
│ - Rate limit check      │
│ - Tenant validation     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ EmbedderService         │
│ - Preprocess text       │
│ - Load tokenizer (L1)   │
│ - Tokenize              │
│ - Load ONNX model (L2)  │
│ - Generate embeddings   │
│ - Pool vectors          │
│ - Format response       │
└────────────┬────────────┘
             │
             ▼
HTTP Response (embeddings)
```

### Text Summarization Flow

```
HTTP Request (text + model)
    │
    ▼
┌─────────────────────────┐
│ Router: /summarize      │
│ - Parse request         │
│ - Validate text size    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ PromptProcessor         │
│ - Load special tokens   │
│ - Get vocab size        │
│ - Create ONNX session   │
│ - Prepare input         │
│ - Run inference         │
│ - Decode output         │
│ - Return summary        │
└────────────┬────────────┘
             │
             ▼
HTTP Response (summary)
```

---

## Security Model

### Authentication & Authorization

```
┌──────────────────────────────────────┐
│    Bearer Token in Authorization     │
│           Header                     │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  AuthMiddleware.verify_token()       │
│  - Decode JWT or API key             │
│  - Verify signature                  │
│  - Check expiration                  │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  TenantMiddleware.verify_tenant()    │
│  - Extract tenant from token         │
│  - Verify tenant code header         │
│  - Load tenant config                │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Request.state.user_id = <id>       │
│  Request.state.tenant_code = <code> │
└──────────────────────────────────────┘
```

### Data Isolation

**Per-Request Isolation**:
- Token contains user ID
- Header contains tenant code
- All database queries filtered by tenant
- Cache keys include tenant code

**Encryption at Rest**:
- Sensitive config values encrypted with Fernet
- Encryption key stored in `.encryption_key` file (0600 permissions)
- Alternative: environment variable `FLOUDS_ENCRYPTION_KEY`

### Rate Limiting

```
Client IP:192.168.1.1 ──► RateLimitMiddleware
                            │
                            ├─ Check request history
                            ├─ Count requests in last 60s
                            ├─ Count requests in last 3600s
                            │
                            ├─ If exceeds per_minute → 429
                            ├─ If exceeds per_hour   → 429
                            └─ Else → Allow request
```

---

## Performance Characteristics

### Latency Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Small embedding (< 100 tokens) | < 100ms | ✅ Good |
| Batch embedding (100 items) | < 500ms | ⚠️ Needs async |
| Text summarization | < 2s | ⚠️ Depends on input |
| Config lookup | < 10ms | ✅ Good (cached) |

### Throughput Metrics

**Current** (single request):
- Embeddings: ~10-50 per second (depends on model)
- Summaries: 1-5 per second (depends on length)

**Bottlenecks**:
1. ❌ Synchronous batch processing (no concurrency)
2. ❌ O(n) rate limit checking (should be O(log n))
3. ❌ Removed asyncio in embedder (performance loss)

### Memory Profile

**Per Instance**:
- Base: ~500MB (Python + FastAPI)
- ONNX model cache: 200-800MB (depends on models)
- In-memory cache: 100-500MB (configurable)
- Total typical: 1-2GB

**Memory Management**:
- Throttled checks every 5 seconds
- Auto-cleanup when available < 1GB
- LRU eviction for cache entries

---

## Deployment Architecture

### Docker Deployment

```yaml
# Single container
services:
  flouds-ai:
    image: flouds-ai:latest
    ports:
      - "19690:19690"
    volumes:
      - ./onnx:/app/onnx
      - ./data:/app/data
    environment:
      FLOUDS_API_ENV: Production
      FLOUDS_ONNX_ROOT: /app/onnx
    restart: unless-stopped
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flouds-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: flouds-ai
  template:
    metadata:
      labels:
        app: flouds-ai
    spec:
      containers:
      - name: flouds-ai
        image: flouds-ai:latest
        ports:
        - containerPort: 19690
        livenessProbe:
          httpGet:
            path: /health/live
            port: 19690
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 19690
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Environment Configuration

```bash
# Required
FLOUDS_API_ENV=Production|Development
FLOUDS_ONNX_ROOT=/path/to/onnx/models

# Optional
FLOUDS_MAX_BATCH_SIZE=20
FLOUDS_MEMORY_LOW_THRESHOLD_MB=150
FLOUDS_LOG_LEVEL=INFO|DEBUG
FLOUDS_ENCRYPTION_KEY=<base64-encoded-key>
```

---

## Future Improvements

### Planned Refactoring
- [ ] Split `PromptService` into 7 modules
- [ ] Split `EmbedderService` into focused components
- [ ] Re-enable async/await in embedder
- [ ] Implement session pooling
- [ ] Add structured logging with correlation IDs

### Performance Optimizations
- [ ] Proper async batch processing
- [ ] Binary search for rate limiting (O(log n))
- [ ] Session connection pooling
- [ ] Path validation caching (LRU)

### Testing Improvements
- [ ] Integration test suite
- [ ] Performance benchmarks
- [ ] Load testing
- [ ] Security penetration testing

---

**Document Version**: 1.0
**Last Updated**: January 17, 2026
**Maintainer**: Development Team
