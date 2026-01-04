import re
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.app_init import APP_SETTINGS
from app.logger import get_logger
from app.modules.key_manager import key_manager
from app.services.config_service import config_service

logger = get_logger("tenant_security")


def _extract_token(request: Request) -> Optional[str]:
    """Extract bearer token from Authorization header or `token` query param in dev."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    if not APP_SETTINGS.app.is_production:
        return request.query_params.get("token")
    return None


def _cors_preflight(origin_value: Optional[str]) -> Response:
    """Return a 204 preflight response with standard CORS headers for origin_value."""
    allow = origin_value or "*"
    headers = {
        "Access-Control-Allow-Origin": allow,
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return Response(status_code=204, headers=headers)


def _apply_cors_headers(response: Response, origin_value: Optional[str]) -> None:
    """Append standard CORS headers to an existing response in-place."""
    allow = origin_value or "*"
    response.headers["Access-Control-Allow-Origin"] = allow
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"


def _match_pattern(value: Optional[str], pattern: Optional[str]) -> bool:
    """Match a value against a single allowed pattern.

    Supported pattern forms:
    - "*" matches everything
    - entries starting with "re:" are treated as a full regular expression
      (e.g. "re:^.*\\.example\\.com$")
    - entries containing '*' are treated as simple wildcards where '*' -> '.*'
      and the pattern is matched as a fullmatch against the value
    - otherwise exact comparison is used
    """
    if pattern is None:
        return False
    # If value is None we cannot match against patterns
    if value is None:
        return False
    if pattern == "*":
        return True
    if pattern.startswith("re:"):
        try:
            rx = pattern[3:]
            return re.fullmatch(rx, value) is not None
        except re.error:
            logger.exception("Invalid regex pattern in allowed list: %s", pattern)
            return False
    if "*" in pattern:
        # Special-case leading '*.' so '*.example.com' also matches 'example.com'
        if pattern.startswith("*.") and pattern.count("*") == 1:
            domain = pattern[2:]
            try:
                regex = r"(^|.*\.)" + re.escape(domain) + r"$"
                return re.fullmatch(regex, value) is not None
            except re.error:
                logger.exception("Wildcard to regex conversion failed for: %s", pattern)
                return False
        # Escape dots and other regex meta chars, then replace '*' => '.*'
        esc = re.escape(pattern)
        esc = esc.replace(r"\*", ".*")
        try:
            return re.fullmatch(esc, value) is not None
        except re.error:
            logger.exception("Wildcard to regex conversion failed for: %s", pattern)
            return False
    return value == pattern


def _is_allowed(value: Optional[str], allowed_list: List[str]) -> bool:
    """Return True if value matches any entry in allowed_list using patterns.

    `value` should already be normalized (for host: hostname only; for origin:
    full origin string or hostname according to caller).
    """
    if not allowed_list:
        return False
    for pat in allowed_list:
        if _match_pattern(value, pat):
            return True
    return False


class TenantTrustedHostMiddleware(BaseHTTPMiddleware):
    """Validate the Host header against tenant-specific trusted hosts.

    Reads `X-Tenant-Code` header to determine tenant. Falls back to default
    hosts in `APP_SETTINGS.security.trusted_hosts` when tenant-specific entry
    is missing. Supports wildcard patterns and regex entries (prefix `re:`).
    """

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        tenant = request.headers.get("X-Tenant-Code", "")
        try:
            allowed = config_service.get_trusted_hosts(tenant_code=tenant)
            if not allowed:
                # fall back to global configured hosts
                from app.app_init import APP_SETTINGS

                allowed = getattr(APP_SETTINGS.security, "trusted_hosts", ["*"])

            # Host header may contain port; compare only hostname portion
            hostname = (host.split(":")[0] if host else "").lower()
            if not _is_allowed(hostname, [a.lower() for a in allowed]):
                # Host is not in the trusted list. As a final override allow
                # a superadmin-authenticated client to bypass this check. We
                # use the same token extraction helper to avoid duplicating logic.
                try:
                    token = _extract_token(request)
                    if token:
                        client = key_manager.authenticate_client(
                            token, tenant_code=tenant or ""
                        )
                        if (
                            client
                            and getattr(client, "client_type", "") == "superadmin"
                        ):
                            logger.info(
                                "Superadmin bypass: allowing request from host %s for tenant %s",
                                hostname,
                                tenant,
                            )
                            return await call_next(request)
                except Exception:
                    logger.exception(
                        "Error checking superadmin bypass for trusted-host"
                    )

                logger.warning(
                    "Blocked request from untrusted host %s for tenant %s",
                    hostname,
                    tenant,
                )
                return JSONResponse(
                    status_code=403, content={"detail": "Untrusted host", "host": host}
                )
        except Exception:
            logger.exception("Trusted host check failed")
            return JSONResponse(
                status_code=500, content={"detail": "Trusted host check failed"}
            )

        return await call_next(request)


class TenantCorsMiddleware(BaseHTTPMiddleware):
    """Apply tenant-specific CORS headers.

    Reads `X-Tenant-Code` header and consults config_service for `cors_origins`.
    If no tenant-specific origins exist, falls back to `APP_SETTINGS.security.cors_origins`.
    Supports wildcard patterns and regex entries (prefix `re:`) for allowed origins.
    This middleware handles preflight (OPTIONS) and appends appropriate CORS headers.
    """

    async def dispatch(self, request: Request, call_next):
        tenant = request.headers.get("X-Tenant-Code", "")
        try:
            origins = config_service.get_cors_origins(tenant_code=tenant)
            if not origins:
                from app.app_init import APP_SETTINGS

                origins = getattr(APP_SETTINGS.security, "cors_origins", ["*"])

            # Normalize allowed origins set
            allowed_origins = origins if origins else ["*"]

            origin_header = request.headers.get("origin")

            # If origins are restricted and the request has an Origin header
            # not in the allowed list, block the request outright. However,
            # allow same-origin requests where the request Host hostname
            # matches the Origin hostname (convenient for local testing).
            parsed = urlparse(origin_header) if origin_header else None
            origin_host = (parsed.hostname if parsed else None) or origin_header

            host = request.headers.get("host", "")
            host_only = (host.split(":")[0] if host else "").lower()
            origin_host_only = (
                origin_host.split(":")[0] if origin_host else ""
            ).lower()

            # Consider localhost and 127.0.0.1 / [::1] equivalent for local dev
            localhost_aliases = {"localhost", "127.0.0.1", "[::1]"}

            def _same_origin(h1: str, h2: str) -> bool:
                if not h1 or not h2:
                    return False
                if h1 == h2:
                    return True
                if h1 in localhost_aliases and h2 in localhost_aliases:
                    return True
                return False

            # If same-origin by hostname (or localhost aliases), allow and echo Origin for preflight
            if origin_header and _same_origin(host_only, origin_host_only):
                if request.method == "OPTIONS":
                    return _cors_preflight(origin_header)
                response = await call_next(request)
                _apply_cors_headers(response, origin_header)
                return response

            if "*" not in allowed_origins and origin_header:
                # Check both full origin and host patterns
                if not (
                    _is_allowed(origin_header, allowed_origins)
                    or _is_allowed(origin_host, allowed_origins)
                ):
                    # Origin not allowed by CORS patterns. As a fallback, consult
                    # tenant-scoped `trusted_hosts` and global `APP_SETTINGS`.
                    # If the Host is trusted for the tenant, allow the request
                    # (this lets trusted hosts bypass strict CORS pattern checks).
                    try:
                        from app.app_init import APP_SETTINGS

                        # tenant may be empty string for default tenant
                        host = request.headers.get("host", "")
                        host_only = (host.split(":")[0] if host else "").lower()

                        trusted = config_service.get_trusted_hosts(tenant_code=tenant)
                        if not trusted:
                            trusted = getattr(
                                APP_SETTINGS.security, "trusted_hosts", ["*"]
                            )

                        # If host matches any trusted-host pattern, treat as allowed
                        # only for authenticated clients (require token). This avoids
                        # silently allowing unauthenticated cross-origin requests
                        # simply because the Host is in a trusted list.
                        if _is_allowed(host_only, [a.lower() for a in trusted]):
                            # Host is trusted for tenant — allow only if the client
                            # is authenticated (require token). If not authenticated
                            # we fall through and allow superadmin-only bypass later.
                            try:
                                token = _extract_token(request)
                                if token:
                                    client = key_manager.authenticate_client(
                                        token, tenant_code=tenant or ""
                                    )
                                    if client:
                                        logger.info(
                                            "Origin %s blocked by CORS but Host %s is trusted for tenant %s and client authenticated; allowing request",
                                            origin_header,
                                            host_only,
                                            tenant,
                                        )
                                        if request.method == "OPTIONS":
                                            return _cors_preflight(origin_header)
                                        response = await call_next(request)
                                        _apply_cors_headers(response, origin_header)
                                        return response
                            except Exception:
                                logger.exception(
                                    "Error authenticating client during trusted-host CORS fallback"
                                )
                        # Not trusted either: allow superadmin bypass if present,
                        # otherwise block as before.
                        # At this point both CORS origin patterns and trusted-host
                        # checks have failed. Allow a superadmin authenticated
                        # client to bypass as a last resort.
                        try:
                            token = _extract_token(request)
                            if token:
                                client = key_manager.authenticate_client(
                                    token, tenant_code=tenant or ""
                                )
                                if (
                                    client
                                    and getattr(client, "client_type", "")
                                    == "superadmin"
                                ):
                                    logger.info(
                                        "Superadmin bypass: allowing cross-origin request from %s for tenant %s",
                                        origin_header,
                                        tenant,
                                    )
                                    if request.method == "OPTIONS":
                                        return _cors_preflight(origin_header)
                                    response = await call_next(request)
                                    _apply_cors_headers(response, origin_header)
                                    return response
                        except Exception:
                            logger.exception(
                                "Error checking superadmin bypass during CORS flow"
                            )

                        logger.warning(
                            "Blocked cross-origin request from %s for tenant %s",
                            origin_header,
                            tenant,
                        )
                        return JSONResponse(
                            status_code=403,
                            content={
                                "detail": "CORS origin not allowed",
                                "origin": origin_header,
                                "origin_host": origin_host,
                            },
                        )
                    except Exception:
                        logger.exception(
                            "Error evaluating trusted hosts during CORS check"
                        )
                        return JSONResponse(
                            status_code=500, content={"detail": "CORS middleware error"}
                        )

            # Determine value to echo in Access-Control-Allow-Origin
            allow_origin = (
                "*"
                if (not origins or "*" in origins)
                else origin_header or ", ".join(origins)
            )

            # Handle preflight
            if request.method == "OPTIONS":
                return _cors_preflight(allow_origin)

            response = await call_next(request)
            # Append CORS headers
            _apply_cors_headers(response, allow_origin)
            return response
        except Exception:
            logger.exception("CORS middleware error")
            return JSONResponse(
                status_code=500, content={"detail": "CORS middleware error"}
            )
