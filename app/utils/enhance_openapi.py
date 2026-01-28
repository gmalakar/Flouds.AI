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

import os
from typing import Any, Optional, cast

# Try to read application settings when available so server URL can be
# constructed dynamically in deployed environments.
try:
    from app.app_init import APP_SETTINGS
except Exception:
    APP_SETTINGS = cast(Any, None)

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def enhance_openapi_schema(app: FastAPI, server_url: Optional[str] = None) -> dict:
    """Build an enhanced OpenAPI schema for `app`.

    Uses the app's title/version/description when available and adds
    contact/license/servers/tags/examples/security information used by
    the Flouds projects.
    """
    # If the schema was already generated, reuse it only when the cached
    # `servers[0].url` matches the requested `server_url` (if provided).
    if app.openapi_schema:
        if server_url is None:
            return app.openapi_schema
        existing_servers = app.openapi_schema.get("servers", [])
        if existing_servers and existing_servers[0].get("url") == server_url:
            return app.openapi_schema

    openapi_schema = get_openapi(
        title=getattr(app, "title", "Flouds API"),
        version=getattr(app, "version", "1.0.0"),
        description=getattr(app, "description", ""),
        routes=app.routes,
    )

    # Add contact/license metadata. Prefer environment variables when provided
    info = openapi_schema.setdefault("info", {})

    contact_name = os.getenv("FLOUDS_CONTACT_NAME")
    contact_email = os.getenv("FLOUDS_CONTACT_EMAIL")
    contact_url = os.getenv("FLOUDS_CONTACT_URL")
    # Only include contact information if environment variables are provided.
    if contact_name or contact_email or contact_url:
        contact = {}
        if contact_name:
            contact["name"] = contact_name
        if contact_url:
            contact["url"] = contact_url
        if contact_email:
            contact["email"] = contact_email
        info["contact"] = contact

    license_name = os.getenv("FLOUDS_LICENSE_NAME")
    license_url = os.getenv("FLOUDS_LICENSE_URL")
    # Only include license information if environment variables are provided.
    if license_name or license_url:
        lic = {}
        if license_name:
            lic["name"] = license_name
        if license_url:
            lic["url"] = license_url
        info["license"] = lic

    # Add servers for convenience. Prefer configured AppSettings
    # `server.openapi_url` when available; otherwise fall back to the
    # legacy development URL for local testing.
    default_url = "http://localhost:19690"
    # Use either the configured AppSettings `server.openapi_url` (if set)
    # or fall back to the legacy development `default_url`.
    try:
        if APP_SETTINGS and getattr(APP_SETTINGS, "server", None):
            server_url_val = APP_SETTINGS.server.openapi_url
        else:
            server_url_val = None
    except Exception:
        server_url_val = None

    if not server_url_val:
        server_url_val = default_url

    openapi_schema.setdefault(
        "servers",
        [
            {
                "url": server_url_val,
                "description": "API server",
            },
        ],
    )

    # Add tags if missing
    openapi_schema.setdefault(
        "tags",
        [
            {"name": "Health", "description": "Health check endpoints"},
            {
                "name": "Administration",
                "description": "Admin and configuration",
            },
            {
                "name": "Model Information",
                "description": "Model info and metadata",
            },
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

    # Attach custom openapi generator.
    # Cast to Any so static checkers allow assignment to the
    # `openapi` attribute while avoiding use of `setattr` which
    # triggers B010.
    cast(Any, app).openapi = _custom


if __name__ == "__main__":
    # Quick local test
    import json

    from fastapi import FastAPI

    a = FastAPI(title="Flouds API Test")
    setup_enhanced_openapi(a)
    print(json.dumps(a.openapi(), indent=2))
