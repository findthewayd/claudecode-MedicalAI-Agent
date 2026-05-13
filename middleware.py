"""API Gateway middleware: rate limiting, auth, metrics, request logging."""
import time
import os
import logging
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ========== In-memory rate limiter ==========

_rate_window = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # seconds
_rate_max = int(os.environ.get("RATE_LIMIT_MAX", "30"))  # requests per window
_rate_store: dict[str, list[float]] = defaultdict(list)


def _clean_old(now, timestamps):
    cutoff = now - _rate_window
    return [t for t in timestamps if t > cutoff]


def check_rate_limit(key: str) -> bool:
    now = time.time()
    timestamps = _clean_old(now, _rate_store[key])
    _rate_store[key] = timestamps
    if len(timestamps) >= _rate_max:
        return False
    _rate_store[key].append(now)
    return True


# ========== Prometheus metrics ==========

_prometheus_available = False
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
    _prometheus_available = True

    http_requests = Counter(
        "http_requests_total", "Total HTTP requests",
        ["method", "endpoint", "status"]
    )
    http_duration = Histogram(
        "http_request_duration_seconds", "HTTP request duration",
        ["method", "endpoint"]
    )
    llm_duration = Histogram(
        "llm_call_duration_seconds", "LLM call duration",
        ["provider", "model"]
    )
    llm_requests = Counter(
        "llm_requests_total", "Total LLM calls",
        ["provider", "model", "status"]
    )
    rag_duration = Histogram(
        "rag_retrieval_duration_seconds", "RAG retrieval duration"
    )
    kg_queries = Counter(
        "kg_queries_total", "Knowledge graph queries", ["backend"]
    )
    memory_usage = Gauge("app_memory_bytes", "Memory usage in bytes")
    active_requests = Gauge("app_active_requests", "Currently active requests")

except ImportError:
    logger.warning("prometheus_client not installed, metrics disabled")


def record_llm_call(provider: str, model: str, duration: float, success: bool):
    if _prometheus_available:
        llm_duration.labels(provider=provider, model=model).observe(duration)
        llm_requests.labels(provider=provider, model=model,
                            status="success" if success else "error").inc()


def record_rag_retrieval(duration: float):
    if _prometheus_available:
        rag_duration.observe(duration)


def record_kg_query(backend: str):
    if _prometheus_available:
        kg_queries.labels(backend=backend).inc()


def update_memory_usage():
    if _prometheus_available:
        try:
            import psutil
            memory_usage.set(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass


# ========== FastAPI Middleware ==========

class GatewayMiddleware(BaseHTTPMiddleware):
    """API Gateway: rate limit + request logging + metrics."""

    async def dispatch(self, request: Request, call_next):
        # Skip static files
        if request.url.path.startswith("/static") or request.url.path in ("/", "/favicon.ico"):
            return await call_next(request)

        # Rate limit (per IP)
        client_ip = request.client.host if request.client else "unknown"
        if request.url.path.startswith("/ask"):
            if not check_rate_limit(client_ip):
                return self._rate_limited()

        # Track active requests
        if _prometheus_available:
            active_requests.inc()

        start = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start

            if _prometheus_available:
                http_requests.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                http_duration.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)

            logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.2f}s)")
            return response
        except Exception as e:
            duration = time.time() - start
            if _prometheus_available:
                http_requests.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=500
                ).inc()
            logger.error(f"{request.method} {request.url.path} -> 500 ({duration:.2f}s): {e}")
            raise
        finally:
            if _prometheus_available:
                active_requests.dec()
            update_memory_usage()

    def _rate_limited(self):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Please slow down."}
        )


# ========== Prometheus endpoint ==========

def get_metrics():
    if not _prometheus_available:
        return "prometheus_client not installed"
    update_memory_usage()
    return generate_latest(REGISTRY)
