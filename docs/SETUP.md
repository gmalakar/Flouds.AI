# Flouds.Py Development Setup Guide

**Last Updated**: January 17, 2026
**Python Version**: 3.11+
**Platform**: Windows, macOS, Linux

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Environment Configuration](#environment-configuration)
4. [ONNX Models Setup](#onnx-models-setup)
5. [IDE Configuration](#ide-configuration)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 20GB for models and virtual environment
- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 20.04+

### Required Software
```bash
# Verify Python version
python --version  # Should be 3.11 or higher

# Git (for cloning repository)
git --version
```

### Optional but Recommended
- **Docker**: For containerized deployment
- **VS Code**: Recommended IDE
- **Git**: For version control

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/Flouds.Py.git
cd Flouds.Py
```

### Step 2: Create Virtual Environment

**Windows (PowerShell)**:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux (Bash)**:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install production dependencies
pip install -r app/requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

**Expected output**:
```
Successfully installed [package list]
```

### Step 4: Configure Environment Variables

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings (see Environment Configuration section)
```

### Step 5: Download ONNX Models

```bash
# Navigate to scripts directory
cd scripts

# Run model downloader (if available)
python download_models.py

# Or manually download from huggingface.co
# Place .onnx files in: ../onnx/[model-type]/
```

### Step 6: Initialize Database

```bash
# The database initializes automatically on first run
# But you can manually initialize:
python -c "from app.services.config_service import init_db; init_db()"
```

### Step 7: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_auth_middleware.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

**Expected output**:
```
tests/test_embedder_service.py::test_embed_text PASSED [10%]
tests/test_config_service.py::test_load_config PASSED [20%]
...
======================== 27 passed in 2.35s =========================
```

### Step 8: Start Development Server

```bash
# Start with auto-reload
uvicorn app.main:app --reload --port 19690

# Start with specific host
uvicorn app.main:app --host 0.0.0.0 --port 19690

# Start with worker processes
uvicorn app.main:app --workers 4 --port 19690
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:19690 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Started reloader process [12346]
```

**Access the application**:
- API Docs: http://localhost:19690/api/v1/docs
- ReDoc: http://localhost:19690/api/v1/redoc
- Health: http://localhost:19690/health

---

## Environment Configuration

### Configuration File (.env)

Create `.env` file in project root:

```bash
# Application settings
FLOUDS_API_ENV=Development              # Development or Production
FLOUDS_APP_NAME=Flouds AI
FLOUDS_APP_VERSION=1.0.0

# Server configuration
FLOUDS_HOST=0.0.0.0
FLOUDS_PORT=19690
FLOUDS_WORKERS=1

# ONNX Model paths
FLOUDS_ONNX_ROOT=./onnx                # Path to ONNX models directory
FLOUDS_CACHE_SIZE=1000                 # Cache size in MB

# Security settings
FLOUDS_ENCRYPTION_KEY=<base64-key>     # Optional: encryption key for config
FLOUDS_MAX_REQUEST_SIZE=26214400       # Max request size in bytes (25MB)

# Rate limiting
FLOUDS_RATE_LIMIT_ENABLED=true
FLOUDS_REQUESTS_PER_MINUTE=60
FLOUDS_REQUESTS_PER_HOUR=1000

# Logging
FLOUDS_LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
FLOUDS_LOG_FILE=./logs/flouds-ai.log

# Database
FLOUDS_CLIENTS_DB_PATH=./data/clients.db

# Feature flags
FLOUDS_ENABLE_RAG=true
FLOUDS_ENABLE_ADMIN=true

# Memory management
FLOUDS_MEMORY_LOW_THRESHOLD_MB=150     # Trigger cache cleanup
FLOUDS_MEMORY_CHECK_INTERVAL=5         # Check interval in seconds
```

### Environment Variables Explained

| Variable | Default | Description |
|----------|---------|-------------|
| `FLOUDS_API_ENV` | Development | Environment mode (affects logging, security headers) |
| `FLOUDS_ONNX_ROOT` | ./onnx | Path to ONNX models directory |
| `FLOUDS_PORT` | 19690 | HTTP server port |
| `FLOUDS_WORKERS` | 1 | Number of worker processes |
| `FLOUDS_MAX_REQUEST_SIZE` | 26214400 | Maximum request body size in bytes |
| `FLOUDS_RATE_LIMIT_ENABLED` | true | Enable/disable rate limiting |
| `FLOUDS_LOG_LEVEL` | INFO | Logging level |

---

## ONNX Models Setup

### Model Directory Structure

```
onnx/
├── embeddings/
│   ├── all-MiniLM-L6-v2.onnx
│   ├── all-mpnet-base-v2.onnx
│   └── ...
├── summarization/
│   ├── facebook-bart-large-cnn.onnx
│   ├── google-flan-t5-base.onnx
│   └── ...
└── extraction/
    ├── bert-base-uncased.onnx
    └── ...
```

### Downloading Models

**Option 1: Using Script** (if available)
```bash
python scripts/download_models.py --models embeddings,summarization
```

**Option 2: Manual Download**
```bash
# Download from Hugging Face Model Hub
# Example: https://huggingface.co/models?library=onnx

# Place in appropriate directory
# Create directories if needed:
mkdir -p onnx/embeddings
mkdir -p onnx/summarization

# Copy downloaded .onnx files to appropriate directories
```

**Option 3: Using Optimum**
```bash
# Convert models to ONNX format
from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    export=True,
    provider="CUDAExecutionProvider"  # or "CPUExecutionProvider"
)

model.save_pretrained("./onnx/embeddings/all-MiniLM-L6-v2")
```

### Verifying Models

```bash
# Check if models are present
ls -la onnx/embeddings/
ls -la onnx/summarization/

# Verify model integrity
python -c "
import onnxruntime as rt
sess = rt.InferenceSession('onnx/embeddings/all-MiniLM-L6-v2.onnx')
print('✓ Model loaded successfully')
"
```

---

## IDE Configuration

### VS Code Setup

**1. Install Extensions**
```
Python (ms-python.python)
Pylance (ms-python.vscode-pylance)
pytest (littlefoxteam.vscode-python-test-adapter)
Pylint (ms-python.pylint)
Black Formatter (ms-python.black-formatter)
YAML (redhat.vscode-yaml)
```

**2. Create `.vscode/settings.json`**
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
```

**3. Debug Configuration** (`.vscode/launch.json`)
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI Debug",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.main:app",
        "--reload",
        "--port", "19690"
      ],
      "jinja": true,
      "console": "integratedTerminal"
    },
    {
      "name": "Pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/",
        "-v"
      ],
      "console": "integratedTerminal"
    }
  ]
}
```

### PyCharm Setup

**1. Configure Interpreter**
- Settings → Project → Python Interpreter
- Select `.venv` interpreter
- Click gear → Add

**2. Enable Inspections**
- Settings → Editor → Python Integrated Tools
- Default test runner: pytest
- Enable PEP 8 warnings

**3. Code Style**
- Settings → Editor → Code Style → Python
- Set line length: 100
- Use Black formatter

---

## Running the Application

### Development Mode

```bash
# Activate virtual environment
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows

# Run with auto-reload
uvicorn app.main:app --reload --port 19690

# Run with debug logging
FLOUDS_LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

### Production Mode

```bash
# Run with multiple workers
uvicorn app.main:app --host 0.0.0.0 --port 19690 --workers 4

# Or use Gunicorn
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Docker

```bash
# Build image
docker build -t flouds-ai:latest .

# Run container
docker run -d \
  --name flouds-ai \
  -p 19690:19690 \
  -v $(pwd)/onnx:/app/onnx \
  -e FLOUDS_API_ENV=Production \
  flouds-ai:latest

# Check logs
docker logs -f flouds-ai

# Stop container
docker stop flouds-ai
```

---

## Troubleshooting

### Issue: Virtual Environment Not Activating

**Windows**:
```powershell
# Error: "cannot be loaded because running scripts is disabled"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Retry activation
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**:
```bash
# Make sure activate script is executable
chmod +x .venv/bin/activate

# Try again
source .venv/bin/activate
```

---

### Issue: ONNX Models Not Found

```bash
# Check environment variable
echo $FLOUDS_ONNX_ROOT  # Unix
echo $env:FLOUDS_ONNX_ROOT  # PowerShell

# Check directory exists
ls -la ./onnx

# Solution: Download models or set correct path
export FLOUDS_ONNX_ROOT=/path/to/onnx
```

---

### Issue: Out of Memory Error

```bash
# Check available memory
free -h  # Linux
wmic OS get TotalVisibleMemorySize  # Windows

# Solution 1: Reduce batch size
export FLOUDS_MAX_BATCH_SIZE=10

# Solution 2: Enable swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### Issue: Port Already in Use

```bash
# Windows
netstat -ano | findstr :19690
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :19690
kill -9 <PID>

# Solution: Use different port
uvicorn app.main:app --port 19691
```

---

### Issue: Import Errors

```bash
# Verify packages installed
pip list | grep -i fastapi
pip list | grep -i onnx

# Reinstall requirements
pip install --force-reinstall -r app/requirements.txt

# Check Python path
python -m pip list
```

---

### Issue: Tests Failing

```bash
# Run tests with verbose output
pytest tests/ -vv -s

# Run specific test
pytest tests/test_config_service.py::test_load_config -vv

# Check test requirements
pip install -r requirements-dev.txt
```

---

## Next Steps

1. ✅ Set up development environment (above)
2. 📖 Read [ARCHITECTURE.md](./ARCHITECTURE.md) for system design
3. 📚 Explore [API.md](./API.md) for endpoint documentation
4. 🚀 Check [DEPLOYMENT.md](./DEPLOYMENT.md) for production setup
5. 🐛 See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues

---

**Document Version**: 1.0
**Last Updated**: January 17, 2026
**Maintainer**: Development Team
