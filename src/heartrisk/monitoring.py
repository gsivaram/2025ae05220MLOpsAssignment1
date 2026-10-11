"""Structured request logging and Prometheus metrics for the API."""

from __future__ import annotations

import json
import logging
import socket
import sys
import time
import uuid

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-ID"
SERVED_BY_HEADER = "X-Served-By"
# Container / pod name: lets clients (and the k8s verification) see which replica answered.
HOSTNAME = socket.gethostname()

# --- Prometheus metrics -------------------------------------------------------
# `path` is the route template (e.g. /predict), never the raw URL, to keep label
# cardinality bounded.
HTTP_REQUESTS = Counter("heartrisk_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "heartrisk_http_request_duration_seconds", "HTTP request latency", ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
IN_FLIGHT = Gauge("heartrisk_http_requests_in_flight", "Requests currently being served")
PREDICTIONS = Counter("heartrisk_predictions_total", "Predictions served, by predicted label", ["label"])
PREDICTION_PROBA = Histogram(
    "heartrisk_prediction_probability", "Predicted probability of disease",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_INFO = Gauge("heartrisk_model_info", "Currently loaded model (always 1)", ["model_name", "model_version"])


# --- JSON logging -------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to grep locally and to ingest in log stacks."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn's own access log would duplicate ours.
    logging.getLogger("uvicorn.access").disabled = True


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


# --- Middleware ---------------------------------------------------------------
access_log = logging.getLogger("heartrisk.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, times the request, logs it and updates HTTP metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        IN_FLIGHT.inc()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[SERVED_BY_HEADER] = HOSTNAME
            return response
        finally:
            IN_FLIGHT.dec()
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            if path != "/metrics":  # don't let Prometheus scrapes dominate the metrics/logs
                HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
                HTTP_LATENCY.labels(request.method, path).observe(elapsed)
                log_event(access_log, "request", request_id=request_id, method=request.method, path=path,
                          status=status, latency_ms=round(elapsed * 1000, 2),
                          client=request.client.host if request.client else None)


def record_prediction(label: str, probability: float) -> None:
    PREDICTIONS.labels(label).inc()
    PREDICTION_PROBA.observe(probability)
