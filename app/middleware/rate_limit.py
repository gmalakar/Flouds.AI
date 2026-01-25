# =============================================================================
# File: rate_limit.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import time
from collections import defaultdict, deque
from typing import Awaitable, Callable, Deque, Dict, List, Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.logger import get_logger
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.performance_tracker import perf_tracker

logger = get_logger("rate_limit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware with configurable limits."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        cleanup_interval: int = 300,
        max_deque_size: int = 2000,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.max_deque_size = max_deque_size

        # Store request timestamps per IP with size limits
        self.request_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_deque_size)
        )

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
        try:
            return request.client.host if request.client else "unknown"
        except AttributeError:
            return "unknown"

    def cleanup_old_requests(self) -> None:
        """Remove old request records to prevent memory leaks."""
        current_time = time.time()

        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        cutoff_time = current_time - 3600  # 1 hour ago
        ips_to_remove: List[str] = []

        # Batch cleanup operations
        for ip, timestamps in self.request_history.items():
            # Remove timestamps older than 1 hour
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()

            # Mark empty entries for removal
            if not timestamps:
                ips_to_remove.append(ip)

        # Remove empty entries in batch
        for ip in ips_to_remove:
            del self.request_history[ip]

        self.last_cleanup = current_time
        if ips_to_remove:
            logger.debug(
                f"Cleaned up {len(ips_to_remove)} empty IPs, {len(self.request_history)} IPs remaining"
            )

    def is_rate_limited(
        self, ip: str, current_time: float
    ) -> Tuple[bool, str, int, int]:
        """Check if IP is rate limited. Returns (is_limited, message, minute_count, hour_count)."""
        timestamps: Deque[float] = self.request_history[ip]

        # Remove old timestamps in one pass
        hour_ago = current_time - 3600
        while timestamps and timestamps[0] < hour_ago:
            timestamps.popleft()

        # Count requests efficiently using binary search approach
        minute_ago = current_time - 60
        hour_count = len(timestamps)

        # Count minute requests by iterating from the end (most recent)
        minute_count = 0
        for i in range(len(timestamps) - 1, -1, -1):
            if timestamps[i] > minute_ago:
                minute_count += 1
            else:
                break

        if minute_count >= self.requests_per_minute:
            return (
                True,
                f"Rate limit exceeded: {minute_count}/{self.requests_per_minute} requests per minute",
                minute_count,
                hour_count,
            )

        if hour_count >= self.requests_per_hour:
            return (
                True,
                f"Rate limit exceeded: {hour_count}/{self.requests_per_hour} requests per hour",
                minute_count,
                hour_count,
            )

        return False, "", minute_count, hour_count

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
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

        # Periodic cleanup (only if needed) with performance tracking
        if time.time() - self.last_cleanup >= self.cleanup_interval:
            with perf_tracker.track("rate_limit_cleanup"):
                self.cleanup_old_requests()

        # Get client IP and current time once
        client_ip = self.get_client_ip(request)
        current_time = time.time()

        # Check rate limit with optimized counting and performance tracking
        with perf_tracker.track("rate_limit_check"):
            is_limited, message, minute_count, hour_count = self.is_rate_limited(
                client_ip, current_time
            )

        if is_limited:
            logger.warning(
                "Rate limit exceeded for IP %s: %s",
                sanitize_for_log(client_ip),
                sanitize_for_log(message),
            )
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": message,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "detail": message,
                },
                headers={
                    "X-RateLimit-Limit-Minute": str(self.requests_per_minute),
                    "X-RateLimit-Remaining-Minute": "0",
                    "X-RateLimit-Limit-Hour": str(self.requests_per_hour),
                    "X-RateLimit-Remaining-Hour": "0",
                    "Retry-After": "60",
                },
            )

        # Record this request
        timestamps: Deque[float] = self.request_history[client_ip]
        timestamps.append(current_time)

        # Ensure deque doesn't exceed size limit
        if len(timestamps) > self.max_deque_size:
            timestamps.popleft()

        # Process request
        response = await call_next(request)

        # Add rate limit headers using pre-calculated counts
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            max(
                0, self.requests_per_minute - minute_count - 1
            )  # -1 for current request
        )
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            max(0, self.requests_per_hour - hour_count - 1)  # -1 for current request
        )

        return response
