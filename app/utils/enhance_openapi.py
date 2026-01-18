#!/usr/bin/env python3
# =============================================================================
# File: enhance_openapi.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
"""
Enhance the FastAPI OpenAPI schema with richer metadata and examples.

This module provides `enhance_openapi_schema(app)` which builds an
enhanced OpenAPI schema and `setup_enhanced_openapi(app)` which attaches
it to the FastAPI application.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def enhance_openapi_schema(app: FastAPI) -> dict:
    """Build an enhanced OpenAPI schema for `app`.

    Uses the app's title/version/description when available and adds
    contact/license/servers/tags/examples/security information used by
    the Flouds projects.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=getattr(app, "title", "Flouds API"),
        version=getattr(app, "version", "0.0.0"),
        description=getattr(app, "description", ""),
        routes=app.routes,
    )

    # Add contact/license metadata
    info = openapi_schema.setdefault("info", {})
    info.setdefault(
        "contact",
        {
            "name": "Flouds API Support",
            "url": "https://github.com/flouds",
            "email": "support@flouds.com",
        },
    )
    info.setdefault(
        "license",
        {
            "name": "Proprietary",
            "url": "https://github.com/flouds/Flouds.Py/blob/main/LICENSE",
        },
    )

    # Add servers for convenience
    openapi_schema.setdefault(
        "servers",
        [
            {"url": "http://localhost:19690", "description": "Development server"},
        ],
    )

    # Add tags if missing
    openapi_schema.setdefault(
        "tags",
        [
            {"name": "Health", "description": "Health check endpoints"},
            {"name": "Administration", "description": "Admin and configuration"},
            {"name": "Model Information", "description": "Model info and metadata"},
        ],
    )

    # Optionally, projects may inject security schemes here. We avoid
    # adding a global `bearerAuth` scheme by default so callers can
    # control security scheme naming and avoid duplicate schemes when
    # multiple tools modify the schema.

    # Optionally enhance some known endpoints with examples
    paths = openapi_schema.get("paths", {})
    if "/api/v1/vector_store/insert" in paths:
        try:
            post = paths["/api/v1/vector_store/insert"]["post"]
            post.setdefault("summary", "Insert vectors with metadata")
            if "requestBody" in post and "content" in post["requestBody"]:
                content = post["requestBody"]["content"]
                if "application/json" in content:
                    content["application/json"].setdefault(
                        "example",
                        {
                            "tenant_code": "demo_tenant",
                            "model_name": "sentence-transformers",
                            "data": [
                                {
                                    "key": "doc_001",
                                    "chunk": "Example document text.",
                                    "model": "sentence-transformers",
                                    "metadata": {"source": "example"},
                                    "vector": [0.1, 0.2, 0.3],
                                }
                            ],
                        },
                    )
        except Exception:
            # Non-critical enhancement failure should not break app startup
            pass

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_enhanced_openapi(app: FastAPI) -> None:
    """Attach enhanced OpenAPI generator to `app`."""

    def _custom():
        return enhance_openapi_schema(app)

    # Attach custom openapi generator
    app.openapi = _custom


if __name__ == "__main__":
    # Quick local test
    import json

    from fastapi import FastAPI

    a = FastAPI(title="Flouds API Test")
    setup_enhanced_openapi(a)
    print(json.dumps(a.openapi(), indent=2))
