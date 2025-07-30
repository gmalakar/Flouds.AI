# =============================================================================
# File: rate_limit.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.logger import get_logger

logger = get_logger("rate_limit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware with configurable limits."""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        cleanup_interval: int = 300,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

        # Store request timestamps per IP
        self.request_history: Dict[str, Deque[float]] = defaultdict(deque)

        # Cleanup interval
        self.last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client host
        return request.client.host if request.client else "unknown"

    def cleanup_old_requests(self) -> None:
        """Remove old request records to prevent memory leaks."""
        current_time = time.time()

        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        cutoff_time = current_time - 3600  # 1 hour ago

        for ip, timestamps in list(self.request_history.items()):
            # Remove timestamps older than 1 hour
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()

            # Remove empty entries
            if not timestamps:
                del self.request_history[ip]

        self.last_cleanup = current_time
        logger.debug(
            f"Cleaned up rate limit history, {len(self.request_history)} IPs remaining"
        )

    def is_rate_limited(self, ip: str) -> tuple[bool, str]:
        """Check if IP is rate limited."""
        current_time = time.time()
        timestamps = self.request_history[ip]

        # Remove old timestamps
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        while timestamps and timestamps[0] < hour_ago:
            timestamps.popleft()

        # Count requests in last minute and hour
        minute_requests = sum(1 for t in timestamps if t > minute_ago)
        hour_requests = len(timestamps)

        if minute_requests >= self.requests_per_minute:
            return (
                True,
                f"Rate limit exceeded: {minute_requests}/{self.requests_per_minute} requests per minute",
            )

        if hour_requests >= self.requests_per_hour:
            return (
                True,
                f"Rate limit exceeded: {hour_requests}/{self.requests_per_hour} requests per hour",
            )

        return False, ""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""

        # Skip rate limiting for health checks and docs
        if request.url.path.startswith(
            (
                "/api/v1/health",
                "/api/v1/docs",
                "/api/v1/redoc",
                "/api/v1/openapi.json",
                "/docs",
                "/redoc",
                "/openapi.json",
            )
        ):
            return await call_next(request)

        # Periodic cleanup
        self.cleanup_old_requests()

        # Get client IP
        client_ip = self.get_client_ip(request)

        # Check rate limit
        is_limited, message = self.is_rate_limited(client_ip)

        if is_limited:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {message}")
            raise HTTPException(
                status_code=429,
                detail={
                    "success": False,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": message,
                    "retry_after_seconds": 60,
                    "timestamp": current_time,
                },
            )

        # Record this request
        current_time = time.time()
        self.request_history[client_ip].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        timestamps = self.request_history[client_ip]
        minute_ago = current_time - 60
        minute_requests = sum(1 for t in timestamps if t > minute_ago)

        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            max(0, self.requests_per_minute - minute_requests)
        )
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            max(0, self.requests_per_hour - len(timestamps))
        )

        return response
