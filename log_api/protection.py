"""Redis-backed HTTP protection and request metrics helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import os
import time
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import redis.asyncio as redis
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


@dataclass(frozen=True)
class RateLimitConfig:
    requests: int = 30
    window_seconds: float = 60.0
    redis_url: str = ""
    key_prefix: str = "sift:ratelimit"

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        requests = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
        window_seconds = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        redis_url = (os.getenv("REDIS_URL") or "").strip()
        key_prefix = (os.getenv("REDIS_RATE_LIMIT_PREFIX") or "sift:ratelimit").strip()
        if requests <= 0:
            raise RuntimeError("RATE_LIMIT_REQUESTS must be positive")
        if window_seconds <= 0:
            raise RuntimeError("RATE_LIMIT_WINDOW_SECONDS must be positive")
        if not redis_url:
            raise RuntimeError("REDIS_URL must be configured for rate limiting")
        if not key_prefix:
            raise RuntimeError("REDIS_RATE_LIMIT_PREFIX must not be blank")
        return cls(
            requests=requests,
            window_seconds=window_seconds,
            redis_url=redis_url,
            key_prefix=key_prefix,
        )


class RateLimiterUnavailable(RuntimeError):
    """Raised when Redis cannot enforce the configured request limit."""


class RedisRateLimiter:
    """Atomic Redis sliding-window limiter shared across API instances."""

    _SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
  local first = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry_ms = window_ms - (now_ms - tonumber(first[2]))
  local retry_seconds = math.max(1, math.ceil(retry_ms / 1000))
  redis.call('EXPIRE', key, math.ceil(window_ms / 1000))
  return {0, retry_seconds, count}
end

redis.call('ZADD', key, now_ms, member)
redis.call('EXPIRE', key, math.ceil(window_ms / 1000))
return {1, 0, count + 1}
"""

    def __init__(self, config: RateLimitConfig, client: Any | None = None):
        self.config = config
        self.client = client or redis.from_url(config.redis_url, decode_responses=True)

    async def check(self, key: str) -> tuple[bool, int]:
        now_ms = int(time.time() * 1000)
        member = f"{now_ms}:{time.monotonic_ns()}"
        redis_key = f"{self.config.key_prefix}:{key}"
        try:
            result = await self.client.eval(
                self._SCRIPT,
                1,
                redis_key,
                now_ms,
                int(self.config.window_seconds * 1000),
                self.config.requests,
                member,
            )
        except RedisError as exc:
            raise RateLimiterUnavailable("Redis rate limiter is unavailable") from exc

        allowed = bool(int(result[0]))
        retry_after = max(1, int(result[1])) if not allowed else 0
        return allowed, retry_after

    async def close(self) -> None:
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()


class RedisMetricsStore:
    """Persist low-cardinality request metrics in minute-level Redis buckets."""

    def __init__(self, client: Any, prefix: str, retention_seconds: int):
        if not prefix.strip():
            raise ValueError("metrics prefix must not be blank")
        if retention_seconds <= 0:
            raise ValueError("metrics retention must be positive")
        self.client = client
        self.prefix = prefix.strip()
        self.retention_seconds = retention_seconds

    def _bucket_key(self, bucket_start: int) -> str:
        return f"{self.prefix}:minute:{bucket_start}"

    def _totals_key(self) -> str:
        return f"{self.prefix}:totals"

    def _visitor_bucket_key(self, bucket_start: int) -> str:
        return f"{self.prefix}:visitors:minute:{bucket_start}"

    def _visitor_digest(self, subject: str) -> str:
        """Return a stable pseudonymous value for Redis HyperLogLog storage."""

        return hashlib.sha256(
            f"{self.prefix}:visitor:{subject}".encode("utf-8")
        ).hexdigest()

    async def record_visitor(self, subject: str) -> None:
        """Count one authenticated user without storing the raw user ID."""

        subject = subject.strip()
        if not subject:
            return
        now = int(time.time())
        bucket_start = now - (now % 60)
        visitor = self._visitor_digest(subject)
        bucket_key = self._visitor_bucket_key(bucket_start)
        try:
            pipeline = self.client.pipeline(transaction=True)
            pipeline.pfadd(bucket_key, visitor)
            pipeline.expire(bucket_key, self.retention_seconds)
            await pipeline.execute()
        except RedisError as exc:
            raise RateLimiterUnavailable("Redis metrics store is unavailable") from exc

    async def record(self, *, status_code: int, latency_ms: float) -> None:
        now = int(time.time())
        bucket_start = now - (now % 60)
        bucket_key = self._bucket_key(bucket_start)
        total_key = self._totals_key()
        fields = {
            "requests": 1,
            "errors": int(status_code >= 500),
            "rate_limited": int(status_code == 429),
            "payload_rejected": int(status_code == 413),
        }
        try:
            pipeline = self.client.pipeline(transaction=True)
            for key in (bucket_key, total_key):
                for field, value in fields.items():
                    pipeline.hincrby(key, field, value)
                pipeline.hincrbyfloat(key, "latency_sum_ms", latency_ms)
            pipeline.expire(bucket_key, self.retention_seconds)
            await pipeline.execute()
        except RedisError as exc:
            raise RateLimiterUnavailable("Redis metrics store is unavailable") from exc

    async def snapshot(self, window_minutes: int) -> dict[str, Any]:
        if window_minutes <= 0:
            raise ValueError("window_minutes must be positive")
        now = int(time.time())
        current_bucket = now - (now % 60)
        bucket_starts = [
            current_bucket - (offset * 60) for offset in range(window_minutes - 1, -1, -1)
        ]
        try:
            pipeline = self.client.pipeline(transaction=False)
            for bucket_start in bucket_starts:
                pipeline.hgetall(self._bucket_key(bucket_start))
            pipeline.hgetall(self._totals_key())
            visitor_keys = [self._visitor_bucket_key(bucket) for bucket in bucket_starts]
            pipeline.pfcount(*visitor_keys)
            values = await pipeline.execute()
        except RedisError as exc:
            raise RateLimiterUnavailable("Redis metrics store is unavailable") from exc

        total_values = values[-2]
        unique_visitors = int(values[-1])
        series = []
        for bucket_start, value in zip(bucket_starts, values[:-1]):
            requests = int(value.get("requests", 0))
            latency_sum = float(value.get("latency_sum_ms", 0.0))
            series.append(
                {
                    "start": bucket_start,
                    "requests": requests,
                    "errors": int(value.get("errors", 0)),
                    "rate_limited": int(value.get("rate_limited", 0)),
                    "payload_rejected": int(value.get("payload_rejected", 0)),
                    "average_latency_ms": round(latency_sum / requests, 2)
                    if requests
                    else 0.0,
                }
            )

        total_requests = int(total_values.get("requests", 0))
        total_latency = float(total_values.get("latency_sum_ms", 0.0))
        return {
            "status": "ready",
            "window_minutes": window_minutes,
            "series": series,
            "totals": {
                "requests": total_requests,
                "errors": int(total_values.get("errors", 0)),
                "rate_limited": int(total_values.get("rate_limited", 0)),
                "payload_rejected": int(total_values.get("payload_rejected", 0)),
                "average_latency_ms": round(total_latency / total_requests, 2)
                if total_requests
                else 0.0,
                "unique_visitors": unique_visitors,
            },
        }


class RequestMetrics:
    """Process-local counters suitable for structured logs and a health dashboard."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.requests_total = 0
        self.rate_limited = 0
        self.payload_rejected = 0
        self.status_codes: Counter[str] = Counter()
        self.routes: Counter[str] = Counter()

    def record(self, path: str, status_code: int) -> None:
        self.requests_total += 1
        self.status_codes[str(status_code)] += 1
        self.routes[path] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "uptime_seconds": round(max(0.0, time.time() - self.started_at), 3),
            "requests_total": self.requests_total,
            "rate_limited": self.rate_limited,
            "payload_rejected": self.payload_rejected,
            "status_codes": dict(sorted(self.status_codes.items())),
            "routes": dict(sorted(self.routes.items())),
        }


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized requests before they reach authentication or model code."""

    def __init__(self, app: Any, max_bytes: int):
        super().__init__(app)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method not in {"POST", "PUT", "PATCH"}:
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = self.max_bytes + 1
            if declared_size > self.max_bytes:
                metrics = getattr(request.app.state, "request_metrics", None)
                if metrics is not None:
                    metrics.payload_rejected += 1
                return JSONResponse(
                    {"detail": "Request payload is too large"},
                    status_code=413,
                    headers={"X-Max-Request-Bytes": str(self.max_bytes)},
                )

        body = await request.body()
        if len(body) > self.max_bytes:
            metrics = getattr(request.app.state, "request_metrics", None)
            if metrics is not None:
                metrics.payload_rejected += 1
            return JSONResponse(
                {"detail": "Request payload is too large"},
                status_code=413,
                headers={"X-Max-Request-Bytes": str(self.max_bytes)},
            )
        return await call_next(request)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Emit one structured request log and update process-local counters."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics = getattr(request.app.state, "request_metrics", None)
            if metrics is not None:
                metrics.record(request.url.path, 500)
            store = getattr(request.app.state, "metrics_store", None)
            if store is not None:
                try:
                    await store.record(status_code=500, latency_ms=(time.perf_counter() - started) * 1000)
                except Exception:
                    request.app.state.request_logger.exception("Redis metrics recording failed")
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        metrics = getattr(request.app.state, "request_metrics", None)
        if metrics is not None:
            metrics.record(request.url.path, response.status_code)
        store = getattr(request.app.state, "metrics_store", None)
        if store is not None:
            try:
                await store.record(
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                )
            except Exception:
                request.app.state.request_logger.exception("Redis metrics recording failed")
        logger = getattr(request.app.state, "request_logger", None)
        if logger is not None:
            logger.info(
                "http_request method=%s path=%s status=%d latency_ms=%.1f "
                "request_bytes=%s response_bytes=%s",
                request.method,
                request.url.path,
                response.status_code,
                latency_ms,
                request.headers.get("content-length", "unknown"),
                response.headers.get("content-length", "unknown"),
            )
        return response


def client_identifier(request: Request, claims: dict[str, Any] | None) -> str:
    """Prefer a verified Supabase subject; use the client IP for local auth-off mode."""

    if claims:
        subject = claims.get("sub") or claims.get("user_id")
        if subject:
            return f"user:{subject}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def create_rate_limit_dependency(limiter: RedisRateLimiter, auth_dependency: Any):
    """Create a FastAPI dependency that runs after the auth dependency."""

    async def dependency(
        request: Request,
        claims: dict[str, Any] | None = Depends(auth_dependency),
    ) -> dict[str, Any] | None:
        metrics_store = getattr(request.app.state, "metrics_store", None)
        subject = str((claims or {}).get("sub") or (claims or {}).get("user_id") or "")
        if metrics_store is not None and subject:
            try:
                await metrics_store.record_visitor(subject)
            except Exception:
                request.app.state.request_logger.exception(
                    "Redis unique visitor metric recording failed"
                )
        key = client_identifier(request, claims)
        try:
            allowed, retry_after = await limiter.check(key)
        except RateLimiterUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="Rate limiting service is temporarily unavailable",
            ) from exc
        if not allowed:
            metrics = getattr(request.app.state, "request_metrics", None)
            if metrics is not None:
                metrics.rate_limited += 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry later.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limiter.config.requests),
                    "X-RateLimit-Remaining": "0",
                },
            )
        return claims

    return dependency
