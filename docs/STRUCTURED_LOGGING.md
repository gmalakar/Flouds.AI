# Structured Logging

Flouds AI includes comprehensive structured JSON logging for production observability, request correlation, and debugging.

## Overview

The structured logging system provides:
- **JSON output format** for log aggregation tools
- **Request correlation** via X-Request-ID headers
- **Automatic timing** for all requests
- **Context injection** for tenant, user, and request metadata
- **Safe payload sanitization** to prevent sensitive data leakage
- **Configurable levels** and output destinations

---

## Configuration

### Enable JSON Logging

```bash
# Environment variable (recommended)
FLOUDS_LOG_JSON=1

# Or in appsettings.json
{
  "logging": {
    "json_format": true
  }
}
```

### Set Log Level

```bash
# Environment variable
FLOUDS_LOG_LEVEL=INFO

# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Configure Output

```bash
# Log file path
FLOUDS_LOG_FILE=logs/flouds.log

# Console output (default: enabled)
FLOUDS_LOG_CONSOLE=1
```

---

## Log Format

### JSON Structure

All logs are emitted as single-line JSON objects:

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

### Core Fields

| Field | Type | Description | Always Present |
|-------|------|-------------|----------------|
| `timestamp` | string | ISO 8601 timestamp with milliseconds | ✅ |
| `level` | string | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | ✅ |
| `logger` | string | Logger name (e.g., "flouds.request") | ✅ |
| `message` | string | Log message | ✅ |
| `module` | string | Python module name | ✅ |
| `func` | string | Function name | ✅ |
| `line` | integer | Line number in source file | ✅ |

### Context Fields

| Field | Type | Description | When Present |
|-------|------|-------------|--------------|
| `request_id` | string | Unique request identifier for correlation | In request context |
| `tenant_code` | string | Tenant identifier for multi-tenancy | When authenticated |
| `user_id` | string | User identifier | When authenticated |
| `path` | string | HTTP request path | In HTTP requests |
| `method` | string | HTTP method (GET, POST, etc.) | In HTTP requests |
| `duration_ms` | float | Request duration in milliseconds | After request completion |

---

## Request Correlation

### X-Request-ID Header

Every request is assigned a unique request ID:

**Client Provides ID:**
```http
GET /api/v1/health HTTP/1.1
X-Request-ID: my-custom-id-123
```

**Server Echoes ID:**
```http
HTTP/1.1 200 OK
X-Request-ID: my-custom-id-123
```

**Server Generates ID:**
```http
GET /api/v1/health HTTP/1.1
# No X-Request-ID header
```

```http
HTTP/1.1 200 OK
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
```

### Correlation Across Services

Use the same request ID when calling downstream services:

```python
import httpx
from app.logger import request_id_var

# Get current request ID
request_id = request_id_var.get("-")

# Pass to downstream service
async with httpx.AsyncClient() as client:
    response = await client.post(
        "https://downstream-service/api",
        headers={"X-Request-ID": request_id},
        json=data
    )
```

---

## Request Timing

All HTTP requests are automatically timed with microsecond precision:

```json
{
  "timestamp": "2026-01-17T12:34:56.789Z",
  "level": "INFO",
  "message": "POST /api/v1/embedder/embed -> 200 [123.45ms]",
  "method": "POST",
  "path": "/api/v1/embedder/embed",
  "duration_ms": 123.45,
  "request_id": "abc-123"
}
```

**Timing Implementation:**
- Start: Request enters middleware
- End: Response ready to send
- Duration: Calculated using `time.perf_counter()`
- Precision: Rounded to 2 decimal places (milliseconds)

---

## Payload Sanitization

### Automatic Sensitive Data Redaction

The logging system automatically redacts sensitive keys in payloads:

**Redacted Keys:**
- `password`
- `token`
- `secret`
- `api_key`
- `apikey`
- `authorization`
- Any key containing "pass", "pwd", "auth", "token", "key", "secret"

**Example:**

**Original Payload:**
```json
{
  "username": "john.doe",
  "password": "super-secret-123",
  "api_key": "sk-proj-abc123",
  "email": "john@example.com"
}
```

**Sanitized Log:**
```json
{
  "username": "john.doe",
  "password": "***REDACTED***",
  "api_key": "***REDACTED***",
  "email": "john@example.com"
}
```

### Manual Sanitization

Use the sanitization function for custom logging:

```python
from app.middleware.log_context import _sanitize_body

# Sanitize JSON payload
payload = b'{"password": "secret123", "data": "public"}'
safe_payload = _sanitize_body(payload, max_length=200)
logger.info(f"Request body: {safe_payload}")

# Output: {"password": "***REDACTED***", "data": "public"}
```

---

## Context Injection

### Setting Request Context

Context is automatically set by middleware, but can be updated programmatically:

```python
from app.logger import set_request_context

# Set multiple context fields
set_request_context(
    request_id="custom-request-123",
    tenant_code="tenant-001",
    user_id="user-456",
    request_path="/api/v1/custom",
    request_method="POST",
    request_duration=45.23
)

# Context automatically appears in all logs
logger.info("Processing request")
# {"message": "Processing request", "request_id": "custom-request-123", ...}
```

### Accessing Context

Retrieve context values in your code:

```python
from app.logger import request_id_var, tenant_code_var, user_id_var

# Get current request ID
request_id = request_id_var.get("-")  # Default: "-"

# Get tenant code
tenant = tenant_code_var.get("-")

# Get user ID
user = user_id_var.get("-")
```

---

## Log Aggregation

### Elasticsearch / ELK Stack

**Filebeat Configuration:**
```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /app/logs/flouds.log
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "flouds-logs-%{+yyyy.MM.dd}"
```

**Kibana Index Pattern:**
- Pattern: `flouds-logs-*`
- Time field: `timestamp`

**Useful Queries:**
```
# All errors for a specific request
request_id: "abc-123" AND level: "ERROR"

# Slow requests (> 1 second)
duration_ms: [1000 TO *]

# Requests from specific tenant
tenant_code: "tenant-001"
```

### CloudWatch Logs

**Log Group Structure:**
```
/aws/flouds-ai/production
  ├── application.log  (JSON logs)
  └── error.log        (ERROR level only)
```

**CloudWatch Insights Queries:**
```sql
-- Average request duration by endpoint
fields @timestamp, path, duration_ms
| filter method = "POST"
| stats avg(duration_ms) by path

-- Error rate by tenant
fields @timestamp, tenant_code, level
| filter level = "ERROR"
| stats count() by tenant_code

-- Request correlation
fields @timestamp, request_id, message
| filter request_id = "abc-123"
| sort @timestamp asc
```

### Splunk

**Index Configuration:**
```ini
[flouds_logs]
sourcetype = _json
INDEXED_EXTRACTIONS = json
KV_MODE = json
TIME_PREFIX = "timestamp":"
TIME_FORMAT = %Y-%m-%dT%H:%M:%S.%3NZ
```

**Search Examples:**
```spl
index=flouds_logs level=ERROR | stats count by module

index=flouds_logs | where duration_ms > 1000 | table timestamp path duration_ms

index=flouds_logs request_id="abc-123" | sort timestamp
```

---

## Performance Considerations

### Log Volume

**Typical Volume:**
- ~1-2 KB per request log entry
- ~5-10 KB per error with stack trace
- ~100 MB/day per 1M requests

**Optimization:**
```bash
# Production: INFO level (recommended)
FLOUDS_LOG_LEVEL=INFO

# High traffic: WARNING level
FLOUDS_LOG_LEVEL=WARNING

# Debug: DEBUG level (temporary only)
FLOUDS_LOG_LEVEL=DEBUG
```

### Log Rotation

Configure log rotation to prevent disk space issues:

```python
# In logging configuration
{
  "handlers": {
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "filename": "logs/flouds.log",
      "maxBytes": 10485760,  # 10MB
      "backupCount": 5
    }
  }
}
```

Or use external tools like `logrotate`:

```bash
# /etc/logrotate.d/flouds
/app/logs/flouds.log {
  daily
  rotate 7
  compress
  delaycompress
  missingok
  notifempty
  create 0644 flouds flouds
}
```

---

## Troubleshooting

### Logs Not Appearing

**Check Configuration:**
```python
import logging
from app.logger import logger

# Verify logger level
print(logger.level)  # Should be <= desired level

# Verify handlers
print(logger.handlers)  # Should include console/file handlers
```

**Check Environment:**
```bash
# Ensure JSON logging is enabled
echo $FLOUDS_LOG_JSON  # Should be "1"

# Check log file permissions
ls -l logs/flouds.log

# Check disk space
df -h /app/logs
```

### Context Not Appearing

**Verify Middleware Registration:**
```python
# In app/app_routing.py
from app.middleware.log_context import LogContextMiddleware

app.add_middleware(LogContextMiddleware)  # Must be registered
```

**Check Context Variables:**
```python
from app.logger import request_id_var

# Should return current request ID, not None
print(request_id_var.get("-"))
```

### Performance Issues

**Reduce Log Volume:**
```bash
# Increase log level
FLOUDS_LOG_LEVEL=WARNING

# Disable debug logging
FLOUDS_LOG_DEBUG=0
```

**Optimize Formatting:**
- Use async logging (if available)
- Configure log buffering
- Consider sampling high-volume endpoints

---

## Best Practices

### DO ✅

- **Use structured fields** instead of string interpolation
  ```python
  # Good
  logger.info("User login", extra={"user_id": user.id, "method": "oauth"})

  # Better (with context)
  set_request_context(user_id=user.id)
  logger.info("User login")
  ```

- **Leverage request correlation**
  ```python
  request_id = request_id_var.get("-")
  logger.info(f"Processing step 1", extra={"request_id": request_id})
  ```

- **Log at appropriate levels**
  - DEBUG: Detailed diagnostic information
  - INFO: General informational messages
  - WARNING: Warning messages (e.g., deprecated API)
  - ERROR: Error events (recoverable)
  - CRITICAL: Critical errors (service down)

### DON'T ❌

- **Don't log sensitive data**
  ```python
  # Bad
  logger.info(f"Password: {password}")

  # Good
  logger.info("Authentication attempt", extra={"user_id": user.id})
  ```

- **Don't log in tight loops**
  ```python
  # Bad
  for item in large_list:
      logger.debug(f"Processing {item}")  # Spam!

  # Good
  logger.info(f"Processing batch", extra={"count": len(large_list)})
  ```

- **Don't mix structured and unstructured**
  ```python
  # Bad
  logger.info(f"User {user_id} logged in")  # String interpolation

  # Good
  logger.info("User logged in", extra={"user_id": user_id})
  ```

---

## Examples

### Request Logging

```python
from fastapi import Request
from app.logger import logger, set_request_context

@app.post("/api/v1/process")
async def process_data(request: Request, data: dict):
    # Context automatically set by middleware

    logger.info("Processing started")
    # Output: {..., "message": "Processing started", "request_id": "abc-123", ...}

    try:
        result = await process(data)
        logger.info("Processing completed", extra={"records": len(result)})
        return result
    except Exception as e:
        logger.error("Processing failed", exc_info=True)
        raise
```

### Background Task Logging

```python
from app.logger import logger, set_request_context
import uuid

async def background_task():
    # Set context for background task
    task_id = str(uuid.uuid4())
    set_request_context(request_id=f"bg-{task_id}")

    logger.info("Background task started")
    # All subsequent logs include task_id

    try:
        await process_large_dataset()
        logger.info("Background task completed")
    except Exception as e:
        logger.error("Background task failed", exc_info=True)
```

### Multi-Step Processing

```python
from app.logger import logger, request_id_var

async def multi_step_process(data):
    request_id = request_id_var.get("-")

    logger.info("Step 1: Validation", extra={"step": 1})
    validated = validate(data)

    logger.info("Step 2: Transformation", extra={"step": 2, "records": len(validated)})
    transformed = transform(validated)

    logger.info("Step 3: Persistence", extra={"step": 3})
    await save(transformed)

    logger.info("Multi-step process completed", extra={"total_steps": 3})
```

---

## Related Documentation

- [Architecture Overview](ARCHITECTURE.md) - System architecture and data flows
- [Environment Variables](ENVIRONMENT.md) - Configuration reference
- [Security Fixes](SECURITY_FIX_SUMMARY.md) - Security features and log sanitization
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md) - Production deployment guide
