# Requirements Analysis - Development vs Production

**Analysis Date**: January 17, 2026
**Status**: ✅ PROPERLY ORGANIZED

---

## Package Classification Analysis

### ✅ Requirements-Dev.txt - ALL DEVELOPMENT-ONLY
**Verdict**: No packages need to move to production

| Package | Category | Purpose | Should Move? |
|---------|----------|---------|--------------|
| pytest, pytest-asyncio, pytest-mock, pytest-timeout, pytest-cov, pytest-benchmark | Testing | Unit & integration testing | ❌ Dev-only |
| black, isort, flake8, flake8-bugbear, flake8-comprehensions | Code Formatting & Linting | Code quality checks | ❌ Dev-only |
| mypy, pyright | Type Checking | Static type analysis | ❌ Dev-only |
| types-requests, types-cryptography, types-setuptools | Type Stubs | Type checking support | ❌ Dev-only |
| bandit, safety | Security Scanning | Vulnerability detection | ❌ Dev-only* |
| interrogate | Documentation | Docstring coverage | ❌ Dev-only |
| pre-commit | Git Hooks | Automated checks on commit | ❌ Dev-only |
| openapi-core | API Validation | OpenAPI spec validation | ❌ Dev-only |

**\* Note on security tools**: While `bandit` and `safety` could theoretically be used in CI/CD pipelines at runtime, they are development/testing tools, not runtime dependencies. They check code and dependencies but don't execute at application runtime.

---

## ✅ Requirements-Prod.txt - CURRENT STATE

**Total Packages**: 19 production dependencies

**Breakdown**:
- **Server** (3): FastAPI, Uvicorn, python-multipart
- **ML/ONNX** (3): Transformers, Optimum, NumPy
- **Runtime Validation** (1): Pydantic
- **System Monitoring** (1): psutil
- **Encryption** (1): cryptography
- **Document Processing** (5): pdfplumber, python-docx, beautifulsoup4, python-pptx, openpyxl
- **NLP** (1): NLTK
- **Utilities** (2): python-dotenv, httpx
- **ONNX Runtime** (1): onnxruntime

---

## Analysis Conclusion

### ✅ Current Organization is CORRECT

**No packages need to be moved between files.**

**Reasoning**:

1. **Development-only Tools**: pytest, black, isort, flake8, mypy, bandit, etc. are only used during development and testing. They have no role at runtime.

2. **Type Stubs Not Needed at Runtime**: Type stub packages (types-*) are only used by type checkers (mypy, pyright) during development. Python doesn't use them at runtime.

3. **Pre-commit and OpenAPI-core**: These are development tools for checking code quality and API specs before commits. Not runtime dependencies.

4. **Security Tools**: While named security packages, `bandit` and `safety` are development tools that analyze code and dependencies. They're not runtime dependencies.

5. **All Runtime Dependencies Already in Production**: The production requirements include everything needed to actually run the application:
   - Web framework (FastAPI, Uvicorn)
   - ML inference (Transformers, ONNX, Optimum)
   - Data processing (Pydantic, NLTK, document loaders)
   - System utilities (psutil, cryptography, dotenv, httpx)

---

## Verification Checklist

- [x] No development tools in requirements-prod.txt
- [x] No runtime dependencies missing from requirements-prod.txt
- [x] Single source of truth (app/requirements.txt references app/requirements-prod.txt)
- [x] Type stubs properly classified as dev-only
- [x] Testing packages properly classified as dev-only
- [x] No duplication between files

---

## Installation Instructions

**Production Only**:
```bash
pip install -r app/requirements-prod.txt
```

**Development** (includes all production + dev tools):
```bash
pip install -r requirements-dev.txt
```

---

**Status**: ✅ OPTIMAL - No changes needed
