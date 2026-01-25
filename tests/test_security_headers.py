# =============================================================================
# File: test_security_headers.py
# Date: 2026-01-17
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_security_headers.py
# Date: 2026-01-17
# =============================================================================

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.security_headers import SecurityHeadersMiddleware


def create_app_with_security(is_production: bool) -> TestClient:
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    app.add_middleware(SecurityHeadersMiddleware, is_production=is_production)
    return TestClient(app)


class TestSecurityHeaders:
    def test_common_security_headers_present(self):
        client = create_app_with_security(is_production=False)
        resp = client.get("/ping")
        assert resp.status_code == 200
        headers = resp.headers
        # Common headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("X-XSS-Protection") == "1; mode=block"
        assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "Permissions-Policy" in headers
        assert "Content-Security-Policy" in headers

    def test_production_headers_stricter(self):
        prod_client = create_app_with_security(is_production=True)
        dev_client = create_app_with_security(is_production=False)

        prod_resp = prod_client.get("/ping")
        dev_resp = dev_client.get("/ping")

        assert prod_resp.status_code == 200
        assert dev_resp.status_code == 200

        prod_csp = prod_resp.headers.get("Content-Security-Policy", "")
        dev_csp = dev_resp.headers.get("Content-Security-Policy", "")

        # Strict-Transport-Security only in production
        assert "Strict-Transport-Security" in prod_resp.headers
        assert "Strict-Transport-Security" not in dev_resp.headers

        # CSP exists in both; production policy should include connect-src 'self'
        assert "default-src" in prod_csp
        assert "default-src" in dev_csp
