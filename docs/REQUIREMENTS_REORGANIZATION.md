# Requirements Files Reorganization - Summary

**Date**: January 17, 2026
**Action**: Separated production and development dependencies

---

## ✅ Changes Made

### requirements-prod.txt (app/requirements-prod.txt)
**Added 2 packages** (previously in requirements-dev.txt):

```
# Environment configuration
python-dotenv>=1.0.0

# HTTP client (for service communication)
httpx>=0.23.0
```

**Rationale**:
- `python-dotenv`: Used in production to load environment variables
- `httpx`: Used in production for HTTP requests to other services

**Total Production Packages**: 17
- 3 Core server packages
- 3 ML/ONNX packages
- 1 Data validation
- 1 System monitoring
- 1 Cryptography
- 5 Document loaders
- 2 Production utilities (newly added)

---

### requirements-dev.txt
**Removed 2 packages** (moved to production):
- `python-dotenv>=1.0.0` → app/requirements-prod.txt
- `httpx>=0.23.0` → app/requirements-prod.txt

**Now contains only development/testing tools**:

| Category | Packages |
|----------|----------|
| Testing (6) | pytest, pytest-asyncio, pytest-mock, pytest-timeout, pytest-cov, pytest-benchmark |
| Code Quality (8) | black, isort, flake8, flake8-bugbear, flake8-comprehensions, mypy, pyright, types-all |
| Security (2) | bandit, safety |
| Documentation (1) | interrogate |
| Pre-commit (1) | pre-commit |
| API Testing (1) | openapi-core |

**Total Development Packages**: 19 (dev-only) + 17 (prod via -r app/requirements.txt)

---

## Installation Instructions

### Production Only
```bash
pip install -r app/requirements-prod.txt
```

### Development (Includes Production)
```bash
pip install -r requirements-dev.txt
```

---

## Dependency Management Best Practices Now Applied

✅ **Separation of Concerns**
- Production packages clearly separated
- Development tools isolated
- No unnecessary bloat in production images

✅ **Explicit Dependencies**
- Production needs clearly documented
- Easy to understand what's required for runtime
- Easy to audit for security issues

✅ **Docker Optimization**
- Production Dockerfile can use only app/requirements-prod.txt
- Smaller Docker image size
- Faster deployment

✅ **Documentation**
- Each package section commented with purpose
- Clear rationale for inclusion

---

## Docker Optimization Example

**Before** (less optimal):
```dockerfile
RUN pip install -r requirements-dev.txt  # 25+ packages
```

**After** (optimized):
```dockerfile
# Production: 17 packages only
RUN pip install -r app/requirements-prod.txt

# Or with dev: 36 total packages (for CI/CD)
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
```

---

## Deployment Checklist

- [x] Production packages properly identified
- [x] Development-only packages separated
- [x] Environment utilities in production (python-dotenv)
- [x] HTTP client in production (httpx)
- [x] Documentation updated
- [x] Installation instructions verified

---

## Files Modified

| File | Changes | Status |
|------|---------|--------|
| app/requirements-prod.txt | Added 2 packages | ✅ |
| requirements-dev.txt | Removed 2 packages, reorganized | ✅ |

---

**Next Steps**: Ready for Phase B (Refactoring & Testing)
