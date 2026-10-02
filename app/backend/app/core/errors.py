from __future__ import annotations
import re
import urllib.parse
from typing import Any

# Standard safe error category constants
SAFE_ERROR_INPUT_VALIDATION = "Invalid input data or EDF recording"
SAFE_ERROR_PROCESSING = "Signal processing failed"
SAFE_ERROR_INFERENCE = "Model inference failed"
SAFE_ERROR_STORAGE = "Storage access failure"
SAFE_ERROR_QUEUE_DISPATCH = "Distributed queue dispatch failed"
SAFE_ERROR_INTERNAL = "Internal processing error"
SAFE_ERROR_LEASE_EXPIRED = "Worker lease expired: distributed worker became unavailable"
SAFE_ERROR_GENERIC = "Job failed during processing"

SAFE_JOB_ERRORS: set[str] = {
    SAFE_ERROR_INPUT_VALIDATION,
    SAFE_ERROR_PROCESSING,
    SAFE_ERROR_INFERENCE,
    SAFE_ERROR_STORAGE,
    SAFE_ERROR_QUEUE_DISPATCH,
    SAFE_ERROR_INTERNAL,
    SAFE_ERROR_LEASE_EXPIRED,
    SAFE_ERROR_GENERIC,
    "Prediction pipeline failed",
}

# Regex to detect credential-bearing URLs
_URL_CREDENTIALS_REGEX = re.compile(r"://([^:@\s]+):([^@\s]+)@")
_URL_PASSWORD_ONLY_REGEX = re.compile(r"://:([^@\s]+)@")

# Regex to detect file paths and extensions
_PATH_INDICATORS = ("/", "\\", ".npz", ".edf", ".parquet", ".csv", ".json", ".txt", "canary")


def mask_redis_url(url: str | None) -> str:
    """Safely mask password/credentials in Redis connection URL."""
    if not url:
        return ""
    try:
        split = urllib.parse.urlsplit(url)
        if split.password:
            user = split.username or ""
            netloc = f"{user}:***@{split.hostname}"
            if split.port:
                netloc += f":{split.port}"
            return urllib.parse.urlunsplit((split.scheme, netloc, split.path, split.query, split.fragment))
    except Exception:
        pass
    # Fallback regex masking
    cleaned = _URL_CREDENTIALS_REGEX.sub(r"://\1:***@", url)
    cleaned = _URL_PASSWORD_ONLY_REGEX.sub(r"://:***@", cleaned)
    return cleaned


_BEARER_TOKEN_REGEX = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
_JWT_REGEX = re.compile(r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b")
_PASSWORD_FIELD_REGEX = re.compile(r"(['\"]?password['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])", re.IGNORECASE)


def sanitize_for_log(text: str | None) -> str:
    """Mask credentials and internal sensitive details for safe server-side logging."""
    if not text:
        return ""
    masked = mask_redis_url(str(text))
    masked = _BEARER_TOKEN_REGEX.sub(r"\1[MASKED_TOKEN]", masked)
    masked = _JWT_REGEX.sub(r"[MASKED_JWT]", masked)
    masked = _PASSWORD_FIELD_REGEX.sub(r"\1***\2", masked)
    return masked


_TECHNICAL_EXCEPTION_REGEX = re.compile(
    r"\b(typeerror|valueerror|ioerror|oserror|runtimeerror|keyerror|indexerror|"
    r"connectionerror|connectionrefusederror|timeouterror|filenotfounderror|"
    r"permissionerror|attributeerror|importerror|modulenotfounderror|assertionerror)\b",
    re.IGNORECASE,
)
_STACK_TRACE_LINE_REGEX = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
_CANARY_REGEX = re.compile(r"\bcanary\b", re.IGNORECASE)
_TECHNICAL_EXCEPTION_INDICATORS = (
    "traceback (most recent call last)",
    "file \"",
    "object at 0x",
    "error:",
    "exception:",
)


def has_path_or_credential_indicators(text: str) -> bool:
    """Check if a string contains internal filesystem paths or URL credentials."""
    if not text:
        return False
    lower = text.lower()
    if any(ind in lower for ind in _PATH_INDICATORS):
        return True
    if "://" in text or "@" in text:
        return True
    return False


def is_unsafe_error_text(text: str | None) -> bool:
    """Check if an error string contains unsafe paths, credentials, URLs, or technical exception traces."""
    if not text:
        return False
    if has_path_or_credential_indicators(text):
        return True
    lower = text.lower()
    if any(ind in lower for ind in _TECHNICAL_EXCEPTION_INDICATORS):
        return True
    if _TECHNICAL_EXCEPTION_REGEX.search(text):
        return True
    if _STACK_TRACE_LINE_REGEX.search(text):
        return True
    if _CANARY_REGEX.search(text):
        return True
    return False


def categorize_exception(exc: Exception) -> str:
    """Map an exception to a safe high-level error category without disclosing internal details."""
    from app.services.storage import StorageError
    from app.services.queue import QueueError

    if isinstance(exc, (StorageError, FileNotFoundError, PermissionError)):
        return SAFE_ERROR_STORAGE
    if isinstance(exc, QueueError):
        return SAFE_ERROR_QUEUE_DISPATCH

    exc_module = getattr(type(exc), "__module__", "").lower()
    exc_name = type(exc).__name__.lower()

    if any(k in exc_module for k in ("mne", "scipy", "pywt")) or "validation" in exc_name:
        return SAFE_ERROR_PROCESSING
    if any(k in exc_module for k in ("lightgbm", "sklearn", "shap")) or "model" in exc_name:
        return SAFE_ERROR_INFERENCE
    if isinstance(exc, (ValueError, TypeError)):
        return SAFE_ERROR_PROCESSING

    return SAFE_ERROR_INTERNAL


def sanitize_error_text(exc_or_text: Exception | str | None, default: str = SAFE_ERROR_INTERNAL) -> str:
    """
    Sanitize an exception message or error string for worker/pipeline storage.
    If the text contains paths, secrets, or internal traces, returns a safe category.
    """
    if exc_or_text is None:
        return default

    msg = str(exc_or_text).strip()
    if not msg:
        return default

    if isinstance(exc_or_text, Exception):
        if is_unsafe_error_text(msg):
            return categorize_exception(exc_or_text)
    else:
        if is_unsafe_error_text(msg):
            return default

    return msg


def sanitize_job_error(error_text: str | None, default: str = SAFE_ERROR_GENERIC) -> str:
    """
    Response-time policy: guarantees that client-visible error fields never
    expose raw exception messages, tracebacks, filesystem paths, or credentials.
    Legacy database rows containing raw errors or technical traces are mapped to safe categories.
    """
    if not error_text or not isinstance(error_text, str) or not error_text.strip():
        return default

    cleaned = error_text.strip()

    # If it is already one of the known safe category strings, return it directly
    if cleaned in SAFE_JOB_ERRORS:
        return cleaned

    # Check if the text contains any unsafe indicators (paths, secrets, URLs, technical exception traces)
    if is_unsafe_error_text(cleaned):
        lower = cleaned.lower()
        # Check for lease expiration
        if "lease expired" in lower or "worker became unavailable" in lower:
            return SAFE_ERROR_LEASE_EXPIRED

        # Check for queue/broker/redis errors
        if "queue" in lower or "broker" in lower or "redis" in lower or "arq" in lower:
            return SAFE_ERROR_QUEUE_DISPATCH

        # Check for storage/file/path errors
        if "storage" in lower or "file" in lower or ".npz" in lower or ".edf" in lower or "not exist" in lower or "not found" in lower:
            return SAFE_ERROR_STORAGE

        # Check for inference/model errors
        if "inference" in lower or "shap" in lower or "lightgbm" in lower or "model" in lower:
            return SAFE_ERROR_INFERENCE

        # Check for signal processing / validation errors
        if "signal" in lower or "processing" in lower or "validation" in lower or "channel" in lower or "mne" in lower:
            return SAFE_ERROR_PROCESSING

        return default

    # If it has no unsafe indicators, it is safe clinical/domain text
    return cleaned


def get_safe_dispatch_error_detail(exc: Exception) -> str:
    """
    Generate client-facing error message for queue dispatch failures.
    Ensures credentials, paths, and internal stack traces are never leaked,
    while preserving existing contract assertions for non-sensitive error messages.
    """
    from app.services.storage import StorageError
    from app.services.queue import QueueError

    if isinstance(exc, QueueError):
        return SAFE_ERROR_QUEUE_DISPATCH

    if isinstance(exc, StorageError):
        msg = str(exc)
        if has_path_or_credential_indicators(msg):
            return f"{SAFE_ERROR_QUEUE_DISPATCH}: {SAFE_ERROR_STORAGE}"
        return f"{SAFE_ERROR_QUEUE_DISPATCH}: {msg}"

    return SAFE_ERROR_QUEUE_DISPATCH
