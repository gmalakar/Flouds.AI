# Token Generation and Usage Guide

## 1. Generate API Token

### Using CLI Script
```bash
# Generate 32-character token
python generate_token.py

# Generate 64-character token
python generate_token.py --length 64

# Generate UUID-based token
python generate_token.py --uuid

# Show environment variable format
python generate_token.py --env
```

### Using Python Code
```python
from app.utils.token_generator import generate_api_token, generate_uuid_token

# Generate secure token
token = generate_api_token(32)
print(f"Token: {token}")

# Generate UUID token
uuid_token = generate_uuid_token()
print(f"UUID Token: {uuid_token}")
```

## 2. Enable Authentication

### Environment Variables
```bash
# Enable security
export FLOUDS_SECURITY_ENABLED=true
export FLOUDS_API_KEY=your_generated_token_here
```

### .env File
```
FLOUDS_SECURITY_ENABLED=true
FLOUDS_API_KEY=your_generated_token_here
```

## 3. Use Token in API Calls

### cURL Examples
```bash
# Summarization
curl -X POST "http://localhost:19690/api/v1/summarizer/summarize" \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{"model": "t5-small", "input": "Text to summarize"}'

# Embedding
curl -X POST "http://localhost:19690/api/v1/embedder/embed" \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{"model": "all-MiniLM-L6-v2", "input": "Text to embed"}'
```

### Python Requests
```python
import requests

token = "your_token_here"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# API call
response = requests.post(
    "http://localhost:19690/api/v1/summarizer/summarize",
    json={"model": "t5-small", "input": "Text to summarize"},
    headers=headers
)
```

## 4. Public Endpoints (No Token Required)

- `/` - Root endpoint
- `/api/v1/health` - Health check
- `/api/v1/docs` - API documentation
- `/api/v1/redoc` - ReDoc documentation
- `/api/v1/openapi.json` - OpenAPI spec