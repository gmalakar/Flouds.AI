# Phase B: Refactoring & Enhancement Plan

**Date**: January 17, 2026
**Objective**: Decompose monolithic services and add integration tests for middleware
**Timeline**: Systematic multi-step refactoring

---

## Overview

Phase B breaks down two large service files (~3,200 lines combined) into focused, maintainable modules while adding integration tests and structured logging.

---

## Part 1: Prompt Service Refactoring (1,731 lines → 7 modules)

### Current Structure Analysis

**Main Class**: `PromptProcessor` (static methods only)

**Responsibilities** (by category):

| Category | Methods | Line Est. |
|----------|---------|----------|
| **Caching/Loading** | `_load_special_tokens`, `_get_special_tokens`, `get_model`, `_get_decoder_session`, `clear_model_cache`, `_check_memory_and_clear_cache` | ~250 |
| **Configuration** | `_get_model_config`, `_get_model_path`, `_get_token_config`, `_get_tokenizer_threadsafe`, `_get_model_file_paths`, `_get_model_filename`, `_get_vocab_size_from_session` | ~350 |
| **Core Processing** | `process_prompt`, `summarize`, `_process_prompt_local`, `summarize_batch_async`, `summarize_batch` | ~200 |
| **Generation Strategies** | `_run_seq2seq_generation`, `_run_encoder_only_generation`, `_run_onnx_generation`, `_generate_seq2seq`, `_generate_decoder_only`, `_generate_onnx` | ~450 |
| **Parameter Building** | `_build_generation_params`, `_add_basic_params`, `_add_beam_params`, `_add_optional_params`, `_add_temperature_params` | ~150 |
| **Text Processing** | `_prepare_input_text`, `_prepend_text`, `_remove_special_tokens`, `_capitalize_sentences` | ~100 |
| **Resource Prep** | `_prepare_model_resources` | ~50 |

### Proposed 7-Module Structure

```
app/services/prompt/
├── __init__.py                    # Public API exports
├── processor.py                   # Main PromptProcessor class (facade)
├── models.py                      # Models: SummaryResults, cached sessions
├── config.py                      # Configuration resolution & token handling
├── generator.py                   # Generation strategies (all _run_* and _generate_* methods)
├── parameters.py                  # Generation parameter building
├── text_utils.py                  # Text preprocessing & post-processing
└── resource_manager.py            # Model/session/cache loading & lifecycle
```

### Module Responsibilities

**`models.py`** (60 lines)
- `SummaryResults` dataclass
- Cached ConcurrentDict instances
- Constants (DEFAULT_MODEL, MEMORY_LOW_THRESHOLD_MB, etc.)

**`resource_manager.py`** (350 lines)
- `_load_special_tokens`
- `_get_decoder_session`
- `_get_special_tokens`
- `get_model`
- `clear_model_cache`
- `_check_memory_and_clear_cache`
- Model/session lifecycle management

**`config.py`** (280 lines)
- `_get_model_config`
- `_get_model_path`
- `_get_token_config`
- `_get_tokenizer_threadsafe`
- `_get_model_file_paths`
- `_get_model_filename`
- `_get_vocab_size_from_session`

**`text_utils.py`** (80 lines)
- `_prepare_input_text`
- `_prepend_text`
- `_remove_special_tokens`
- `_capitalize_sentences`

**`parameters.py`** (120 lines)
- `_build_generation_params`
- `_add_basic_params`
- `_add_beam_params`
- `_add_optional_params`
- `_add_temperature_params`

**`generator.py`** (350 lines)
- `_run_seq2seq_generation`
- `_run_encoder_only_generation`
- `_run_onnx_generation`
- `_generate_seq2seq`
- `_generate_decoder_only`
- `_generate_onnx`
- Helper methods for generation logic

**`processor.py`** (150 lines)
- `PromptProcessor` class (main orchestrator)
- `process_prompt`
- `summarize`
- `_process_prompt_local`
- `_prepare_model_resources`
- `summarize_batch_async`
- `summarize_batch`

**`__init__.py`** (15 lines)
```python
from app.services.prompt.processor import PromptProcessor
from app.services.prompt.models import SummaryResults

__all__ = ["PromptProcessor", "SummaryResults"]
```

### Dependency Graph

```
processor.py (main entry point)
  ├── models.py (constants, cached sessions)
  ├── config.py (resolve model paths, tokenizers)
  ├── text_utils.py (prepare input text)
  ├── parameters.py (build generation params)
  ├── generator.py (run generation)
  │   ├── resource_manager.py (get sessions)
  │   └── text_utils.py (post-process text)
  └── resource_manager.py (load models, manage cache)
```

### Benefits

- **Separation of Concerns**: Each module has single responsibility
- **Testability**: Mock resource_manager.py for unit tests; easier fixtures
- **Maintainability**: 200-350 lines per module vs. 1,731 in one file
- **Reusability**: Other services can import `resource_manager` for caching patterns
- **Scalability**: Adding new generation strategies doesn't touch existing code

---

## Part 2: Embedder Service Refactoring (1,514 lines)

### Current Structure Analysis

**Main Class**: `Embedder` (static/class methods)

**Responsibilities** (estimated):

| Category | Purpose | Lines |
|----------|---------|-------|
| **Model Loading** | Load/cache embedding models | ~250 |
| **Inference** | Encode text, batch processing | ~400 |
| **Caching** | Encoder output cache, special logic | ~200 |
| **Error Handling** | Exception wrapping, validation | ~150 |
| **Utilities** | Tokenization, normalization, text prep | ~250 |
| **Async Operations** | Async batch encoding, concurrent tasks | ~200 |
| **Testing Helpers** | Mocks, test utilities | ~100 |

### Proposed Module Structure

```
app/services/embedder/
├── __init__.py                    # Public API
├── embedder.py                    # Main Embedder class (facade)
├── models.py                      # Data models, cache configs
├── inference.py                   # Core encoding logic
├── caching.py                     # Encoder output caching
├── text_utils.py                  # Text preprocessing
└── async_worker.py                # Async batch processing
```

---

## Part 3: Middleware Integration Tests

### New Test Files

**`tests/test_request_size_limit.py`** (~60 lines)
```python
def test_request_under_limit():
    # Verify normal requests pass through

def test_request_over_limit():
    # Verify 413 Payload Too Large returned

def test_content_length_validation():
    # Test with/without Content-Length header
```

**`tests/test_security_headers.py`** (~80 lines)
```python
def test_security_headers_in_response():
    # Verify headers present (X-Frame-Options, CSP, etc.)

def test_production_vs_dev_headers():
    # Different headers for prod vs dev environment

def test_cors_headers():
    # Verify CORS headers included
```

---

## Part 4: Structured Logging

### Changes to `app/logger.py`

Add support for:
- **Context Variables**: Request ID, tenant code, user ID
- **JSON Formatting**: Structured output for log aggregation
- **Request Tracing**: Track requests through all services

### Example Usage

```python
from app.logger import get_logger, set_request_context

logger = get_logger("prompt_service")
set_request_context(request_id="req-123", tenant="acme")
logger.info("Processing prompt", extra={"model": "t5-small"})
# Output: {"timestamp": "...", "level": "INFO", "request_id": "req-123", ...}
```

---

## Execution Order

### Week 1: Prompt Service
1. **Day 1**: Create `app/services/prompt/` module structure (7 files)
2. **Day 2**: Migrate `resource_manager.py` code
3. **Day 3**: Migrate `config.py` code
4. **Day 4**: Migrate `generator.py` and test
5. **Day 5**: Migrate remaining modules; update imports

### Week 2: Embedder Service
1. **Day 1-3**: Apply similar refactoring to embedder service
2. **Day 4**: Middleware integration tests
3. **Day 5**: Structured logging implementation

---

## Testing Strategy

### Unit Tests
- Mock resource_manager, config loaders
- Test each module independently
- Target 80%+ coverage

### Integration Tests
- Test full pipeline: request → processor → response
- Verify middleware behavior
- Test async batch processing

### Regression Tests
- Run existing tests against refactored code
- Verify API compatibility
- Check performance (no regressions)

---

## Success Criteria

✅ All existing tests pass
✅ New middleware tests added (8+ test cases)
✅ Code coverage maintained or improved
✅ No breaking changes to public APIs
✅ Structured logging enabled across services
✅ Documentation updated (ARCHITECTURE.md)

---

## Rollback Plan

If issues arise:
1. Keep original `prompt_service.py` as `.bak` during refactoring
2. Maintain git commits after each module migration
3. Easy revert to original if needed
