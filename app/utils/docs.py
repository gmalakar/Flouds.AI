# =============================================================================
# File: docs.py
# Date: 2026-01-28
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
import requests  # type: ignore[import-untyped]
from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, Response

from app.app_init import APP_SETTINGS
from app.utils.enhance_openapi import enhance_openapi_schema


def _derive_server_url(request: Request) -> str:
    headers = request.headers
    proto = headers.get("x-forwarded-proto")
    if proto:
        scheme = proto.split(",")[0].strip()
    else:
        scheme = request.url.scheme

    host = headers.get("x-forwarded-host") or headers.get("host") or request.url.netloc
    host = host.rstrip("/")
    return f"{scheme}://{host}"


def _docs_asset_url(request: Request, asset_path: str) -> str:
    use_proxy = getattr(APP_SETTINGS.server, "docs_use_proxy", False)
    if use_proxy:
        server_url = _derive_server_url(request)
        return f"{server_url}/_docs_assets/{asset_path}"

    base = getattr(APP_SETTINGS.server, "docs_asset_base", None)
    if base:
        return f"{base.rstrip('/')}/{asset_path}"

    return f"https://cdn.jsdelivr.net/npm/{asset_path}"


def register_docs_routes(app: FastAPI, api_prefix: str) -> None:
    """Register Swagger / ReDoc UI routes and an optional asset proxy.

    Args:
        app: FastAPI application
        api_prefix: API prefix, e.g. '/api/v1'
    """

    @app.get(f"{api_prefix}/docs", include_in_schema=False)
    async def custom_swagger_ui(request: Request) -> HTMLResponse:
        server_url = _derive_server_url(request)
        enhance_openapi_schema(app, server_url=server_url)
        openapi_url = f"{server_url}{api_prefix}/openapi.json"
        js = _docs_asset_url(request, "swagger-ui-dist@5/swagger-ui-bundle.js")
        css = _docs_asset_url(request, "swagger-ui-dist@5/swagger-ui.css")
        return get_swagger_ui_html(
            openapi_url=openapi_url,
            title=f"{app.title} - Swagger UI",
            swagger_js_url=js,
            swagger_css_url=css,
        )

    @app.get(f"{api_prefix}/redoc", include_in_schema=False)
    async def custom_redoc_ui(request: Request) -> HTMLResponse:
        server_url = _derive_server_url(request)
        enhance_openapi_schema(app, server_url=server_url)
        openapi_url = f"{server_url}{api_prefix}/openapi.json"
        redoc_js = _docs_asset_url(request, "redoc@next/bundles/redoc.standalone.js")
        return get_redoc_html(
            openapi_url=openapi_url,
            title=f"{app.title} - ReDoc",
            redoc_js_url=redoc_js,
        )

    @app.get("/_docs_assets/{path:path}", include_in_schema=False)
    def _docs_assets_proxy(path: str, request: Request) -> Response:
        upstream_base = getattr(
            APP_SETTINGS.server, "docs_asset_base", "https://cdn.jsdelivr.net/npm"
        )
        upstream_url = f"{upstream_base.rstrip('/')}/{path}"
        try:
            r = requests.get(upstream_url, timeout=10)
            content = r.content
            media = r.headers.get("content-type", "application/octet-stream")
            headers = {"cache-control": "public, max-age=3600"}
            return Response(content=content, media_type=media, headers=headers)
        except Exception:
            return Response(status_code=502)
