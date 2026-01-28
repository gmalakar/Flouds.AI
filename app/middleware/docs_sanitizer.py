# =============================================================================
# File: docs_sanitizer.py
# Date: 2026-01-27
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

import re
from typing import Any, Callable, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

CF_BEACON_RE = re.compile(
    r"<script[^>]*static\.cloudflareinsights\.com[^>]*>.*?</script>", re.I | re.S
)


class DocsSanitizerMiddleware(BaseHTTPMiddleware):
    """Remove known telemetry/script snippets from docs HTML responses.

    This middleware is intentionally narrow: it only inspects HTML responses
    for docs paths and strips occurrences of the Cloudflare Insights beacon
    script. It avoids changing binary/JSON responses.
    """

    def __init__(self, app: Any, docs_paths: Optional[List[str]] = None) -> None:
        super().__init__(app)
        self.docs_paths = docs_paths or ["/api/v1/docs", "/api/v1/redoc", "/docs", "/redoc"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Only operate on HTML responses for docs paths
        path = request.url.path
        content_type = response.headers.get("content-type", "")
        if any(path.startswith(p) for p in self.docs_paths) and "html" in content_type.lower():
            # Consume body iterator and reconstruct response after substitution
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Determine new content (string) after removing known telemetry
            try:
                text = body.decode(response.charset or "utf-8", errors="ignore")
                new_text = CF_BEACON_RE.sub("", text)
            except Exception:
                new_text = None

            # Build headers copy and remove Content-Length to avoid mismatch
            headers = dict(response.headers)
            for hk in list(headers.keys()):
                if hk.lower() == "content-length":
                    del headers[hk]

            # If we produced a modified text, return it as HTML string; otherwise
            # return the original bytes with the original content type.
            if new_text is not None and new_text != text:
                return Response(
                    content=new_text,
                    status_code=response.status_code,
                    headers=headers,
                    media_type="text/html",
                )
            else:
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=content_type,
                )

        return response
