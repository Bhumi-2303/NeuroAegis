from __future__ import annotations
import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

# ContextVars for request and execution correlation
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_ctx: ContextVar[str | None] = ContextVar("job_id", default=None)
tenant_id_ctx: ContextVar[str | None] = ContextVar("tenant_id", default=None)
worker_id_ctx: ContextVar[str | None] = ContextVar("worker_id", default=None)

_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def validate_or_generate_request_id(header_val: str | None) -> str:
    """
    Validate an incoming X-Request-ID header value.
    Accepts bounded alphanumeric identifiers with hyphens/underscores up to 64 chars.
    Rejects control characters, invalid characters, or oversized strings and returns a fresh UUIDv4.
    """
    if header_val and isinstance(header_val, str):
        cleaned = header_val.strip()
        if _REQUEST_ID_REGEX.match(cleaned):
            return cleaned
    return str(uuid.uuid4())


def get_request_id() -> str | None:
    return request_id_ctx.get()


def get_job_id() -> str | None:
    return job_id_ctx.get()


def get_tenant_id() -> str | None:
    return tenant_id_ctx.get()


def get_worker_id() -> str | None:
    return worker_id_ctx.get()


def set_logging_context(
    request_id: str | None = None,
    job_id: str | None = None,
    tenant_id: str | None = None,
    worker_id: str | None = None,
) -> dict[str, Token]:
    """Sets correlation identifiers in ContextVars and returns tokens for cleanup."""
    tokens: dict[str, Token] = {}
    if request_id is not None:
        tokens["request_id"] = request_id_ctx.set(request_id)
    if job_id is not None:
        tokens["job_id"] = job_id_ctx.set(job_id)
    if tenant_id is not None:
        tokens["tenant_id"] = tenant_id_ctx.set(tenant_id)
    if worker_id is not None:
        tokens["worker_id"] = worker_id_ctx.set(worker_id)
    return tokens


def reset_logging_context(tokens: dict[str, Token]) -> None:
    """Resets ContextVars using the returned tokens."""
    if "worker_id" in tokens:
        worker_id_ctx.reset(tokens["worker_id"])
    if "tenant_id" in tokens:
        tenant_id_ctx.reset(tokens["tenant_id"])
    if "job_id" in tokens:
        job_id_ctx.reset(tokens["job_id"])
    if "request_id" in tokens:
        request_id_ctx.reset(tokens["request_id"])


class StructuredJsonFormatter(logging.Formatter):
    """
    Machine-parseable, container-friendly JSON log formatter.
    Safely includes correlation fields (request_id, job_id, tenant_id, worker_id)
    and sanitizes technical strings to prevent credential/path leakage.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()

        # Sanitize message using Phase 10.1 facility
        try:
            from app.core.errors import sanitize_for_log
            safe_message = sanitize_for_log(message)
        except Exception:
            safe_message = message

        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": safe_message,
        }

        # Correlated context fields (always present for structured log schema stability)
        log_entry["request_id"] = getattr(record, "request_id", None) or get_request_id()
        log_entry["job_id"] = getattr(record, "job_id", None) or get_job_id()
        log_entry["tenant_id"] = getattr(record, "tenant_id", None) or get_tenant_id()
        log_entry["worker_id"] = getattr(record, "worker_id", None) or get_worker_id()

        # Exception diagnosis without sensitive leakage
        if record.exc_info and record.exc_info[1]:
            exc = record.exc_info[1]
            try:
                from app.core.errors import categorize_exception, sanitize_error_text
                log_entry["exception_type"] = type(exc).__name__
                log_entry["exception_category"] = categorize_exception(exc)
                log_entry["exception"] = sanitize_error_text(exc)
            except Exception:
                log_entry["exception_type"] = type(exc).__name__

        return json.dumps(log_entry, default=str)


_LOGGING_INITIALIZED = False


def setup_logging(force: bool = False) -> logging.Logger:
    """
    Idempotently configure structured JSON logging for the application and workers.
    Ensures repeated calls do not create duplicate handlers or duplicate log lines.
    """
    global _LOGGING_INITIALIZED

    log_level_name = getattr(settings, "LOG_LEVEL", "info").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Check if StructuredJsonFormatter handler is already installed
    has_json_handler = False
    for handler in root_logger.handlers:
        if isinstance(getattr(handler, "formatter", None), StructuredJsonFormatter):
            has_json_handler = True
            handler.setLevel(log_level)
            break

    if not has_json_handler or force:
        # Clear existing stdout StreamHandlers to prevent duplication
        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(StructuredJsonFormatter())
        root_logger.addHandler(stream_handler)

    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.getLogger("mne").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _LOGGING_INITIALIZED = True
    return logging.getLogger("neuroaegis")


logger = setup_logging()
