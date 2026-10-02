from __future__ import annotations
import math
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

# Bounded label enumerations
ALLOWED_EXECUTION_MODES = {"in_process", "distributed"}
ALLOWED_STATUSES = {"success", "failed"}
ALLOWED_STAGES = {
    "staging_save",
    "queue_wait",
    "worker_claim",
    "staging_load",
    "preprocessing",
    "feature_extraction",
    "model_inference",
    "explanation",
    "persistence",
    "cleanup",
}
ALLOWED_DATASETS = {"bonn", "chbmit", "unknown"}

PROHIBITED_LABEL_KEYS = {
    "request_id",
    "job_id",
    "tenant_id",
    "patient_id",
    "patient_name",
    "token",
    "password",
    "cookie",
    "traceback",
    "error",
}


def _escape_prometheus_string(value: str) -> str:
    """Escape label values according to Prometheus text exposition rules."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def validate_metric_labels(declared_keys: tuple[str, ...], labels: dict[str, str] | None) -> dict[str, str]:
    """
    Validate that provided labels match declared keys exactly and contain only bounded values.
    Raises ValueError if validation fails.
    """
    cleaned: dict[str, str] = {}
    if not declared_keys:
        if labels:
            raise ValueError(f"Metric does not accept labels, but received: {list(labels.keys())}")
        return cleaned

    if labels is None:
        raise ValueError(f"Metric requires labels {declared_keys}, but None was provided")

    given_keys = set(labels.keys())
    declared_set = set(declared_keys)

    if given_keys != declared_set:
        missing = declared_set - given_keys
        extra = given_keys - declared_set
        raise ValueError(f"Label mismatch. Missing: {missing}, Extra: {extra}")

    for k, v in labels.items():
        if k in PROHIBITED_LABEL_KEYS:
            raise ValueError(f"Prohibited high-cardinality or sensitive label key: {k}")

        str_val = str(v).strip()
        if not str_val:
            raise ValueError(f"Empty label value for key: {k}")

        if k == "status" and str_val not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status label value: '{str_val}'. Allowed: {ALLOWED_STATUSES}")
        elif k == "execution_mode" and str_val not in ALLOWED_EXECUTION_MODES:
            raise ValueError(f"Invalid execution_mode label value: '{str_val}'. Allowed: {ALLOWED_EXECUTION_MODES}")
        elif k == "stage" and str_val not in ALLOWED_STAGES:
            raise ValueError(f"Invalid stage label value: '{str_val}'. Allowed: {ALLOWED_STAGES}")
        elif k == "dataset" and str_val not in ALLOWED_DATASETS:
            raise ValueError(f"Invalid dataset label value: '{str_val}'. Allowed: {ALLOWED_DATASETS}")
        elif k == "error_category":
            from app.core.errors import SAFE_JOB_ERRORS
            if str_val not in SAFE_JOB_ERRORS:
                raise ValueError(f"Invalid error_category label value: '{str_val}'. Must be in SAFE_JOB_ERRORS")

        cleaned[k] = str_val

    return cleaned


class Counter:
    """Thread-safe monotonic Prometheus counter."""

    def __init__(self, name: str, description: str, label_keys: tuple[str, ...] = ()):
        self.name = name
        self.description = description
        self.label_keys = tuple(label_keys)
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if amount < 0:
            raise ValueError("Counter increments must be non-negative")
        valid_labels = validate_metric_labels(self.label_keys, labels)
        key = tuple(sorted(valid_labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def get_value(self, labels: dict[str, str] | None = None) -> float:
        valid_labels = validate_metric_labels(self.label_keys, labels)
        key = tuple(sorted(valid_labels.items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def format_prometheus(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            sorted_keys = sorted(self._values.keys())
            for key in sorted_keys:
                val = self._values[key]
                if not key:
                    lines.append(f"{self.name} {val:g}")
                else:
                    label_str = ",".join(f'{k}="{_escape_prometheus_string(v)}"' for k, v in key)
                    lines.append(f"{self.name}{{{label_str}}} {val:g}")
        return lines


class Gauge:
    """Thread-safe Prometheus gauge for instantaneous values."""

    def __init__(self, name: str, description: str, label_keys: tuple[str, ...] = ()):
        self.name = name
        self.description = description
        self.label_keys = tuple(label_keys)
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        valid_labels = validate_metric_labels(self.label_keys, labels)
        key = tuple(sorted(valid_labels.items()))
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        valid_labels = validate_metric_labels(self.label_keys, labels)
        key = tuple(sorted(valid_labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        valid_labels = validate_metric_labels(self.label_keys, labels)
        key = tuple(sorted(valid_labels.items()))
        with self._lock:
            current = self._values.get(key, 0.0)
            self._values[key] = max(0.0, current - float(amount))

    def get_value(self, labels: dict[str, str] | None = None) -> float:
        valid_labels = validate_metric_labels(self.label_keys, labels)
        key = tuple(sorted(valid_labels.items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def format_prometheus(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            sorted_keys = sorted(self._values.keys())
            for key in sorted_keys:
                val = self._values[key]
                if not key:
                    lines.append(f"{self.name} {val:g}")
                else:
                    label_str = ",".join(f'{k}="{_escape_prometheus_string(v)}"' for k, v in key)
                    lines.append(f"{self.name}{{{label_str}}} {val:g}")
        return lines


class Histogram:
    """Thread-safe Prometheus histogram with cumulative buckets, sum, and count."""

    def __init__(
        self,
        name: str,
        description: str,
        label_keys: tuple[str, ...] = (),
        buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    ):
        self.name = name
        self.description = description
        self.label_keys = tuple(label_keys)
        # Ensure buckets are sorted and finite
        self.buckets = tuple(sorted(float(b) for b in buckets if math.isfinite(b)))
        self._lock = threading.Lock()
        self._counts: dict[tuple[tuple[str, str], ...], dict[float, int]] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = {}
        self._totals: dict[tuple[tuple[str, str], ...], int] = {}

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        valid_labels = validate_metric_labels(self.label_keys, labels)
        val = max(0.0, float(value))
        key = tuple(sorted(valid_labels.items()))

        with self._lock:
            if key not in self._counts:
                self._counts[key] = {b: 0 for b in self.buckets}
                self._sums[key] = 0.0
                self._totals[key] = 0

            self._sums[key] += val
            self._totals[key] += 1
            for b in self.buckets:
                if val <= b:
                    self._counts[key][b] += 1

    def get_stats(self, labels: dict[str, str] | None = None) -> dict[str, Any]:
        valid_labels = validate_metric_labels(self.label_keys, labels)
        key = tuple(sorted(valid_labels.items()))
        with self._lock:
            count = self._totals.get(key, 0)
            total_sum = self._sums.get(key, 0.0)
            bucket_counts = dict(self._counts.get(key, {b: 0 for b in self.buckets}))
            return {"count": count, "sum": total_sum, "buckets": bucket_counts}

    def format_prometheus(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            sorted_keys = sorted(self._counts.keys())
            for key in sorted_keys:
                base_labels = [f'{k}="{_escape_prometheus_string(v)}"' for k, v in key]
                # Bucket lines
                for b in self.buckets:
                    b_count = self._counts[key][b]
                    le_label = f'le="{b:g}"'
                    all_labels = ",".join(base_labels + [le_label])
                    lines.append(f"{self.name}_bucket{{{all_labels}}} {b_count}")

                # +Inf bucket
                total_count = self._totals[key]
                all_labels_inf = ",".join(base_labels + ['le="+Inf"'])
                lines.append(f"{self.name}_bucket{{{all_labels_inf}}} {total_count}")

                # Sum and count
                label_suffix = f"{{{','.join(base_labels)}}}" if base_labels else ""
                lines.append(f"{self.name}_sum{label_suffix} {self._sums[key]:.6f}")
                lines.append(f"{self.name}_count{label_suffix} {total_count}")

        return lines


class MetricsRegistry:
    """Central registry of registered Prometheus metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: dict[str, Any] = {}

    def register_counter(self, name: str, description: str, label_keys: tuple[str, ...] = ()) -> Counter:
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            counter = Counter(name, description, label_keys)
            self._metrics[name] = counter
            return counter

    def register_gauge(self, name: str, description: str, label_keys: tuple[str, ...] = ()) -> Gauge:
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            gauge = Gauge(name, description, label_keys)
            self._metrics[name] = gauge
            return gauge

    def register_histogram(
        self,
        name: str,
        description: str,
        label_keys: tuple[str, ...] = (),
        buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    ) -> Histogram:
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            histogram = Histogram(name, description, label_keys, buckets)
            self._metrics[name] = histogram
            return histogram

    def get_metric(self, name: str) -> Any | None:
        with self._lock:
            return self._metrics.get(name)

    def generate_prometheus_text(self) -> str:
        with self._lock:
            all_metrics = sorted(self._metrics.values(), key=lambda m: m.name)

        output_lines: list[str] = []
        for metric in all_metrics:
            output_lines.extend(metric.format_prometheus())

        return "\n".join(output_lines) + "\n"

    def clear(self) -> None:
        """Reset internal metrics storage (used for isolated test suites)."""
        with self._lock:
            for metric in self._metrics.values():
                with metric._lock:
                    if hasattr(metric, "_values"):
                        metric._values.clear()
                    if hasattr(metric, "_counts"):
                        metric._counts.clear()
                        metric._sums.clear()
                        metric._totals.clear()


# Global Singleton Registry
_REGISTRY = MetricsRegistry()


def get_telemetry_registry() -> MetricsRegistry:
    return _REGISTRY


# Standard Required Metrics (Step 4)
JOBS_CREATED_TOTAL = _REGISTRY.register_counter(
    "neuroaegis_jobs_created_total",
    "Total prediction jobs created",
    label_keys=("dataset", "execution_mode"),
)

JOBS_COMPLETED_TOTAL = _REGISTRY.register_counter(
    "neuroaegis_jobs_completed_total",
    "Total prediction jobs completed successfully",
    label_keys=("dataset", "execution_mode"),
)

JOBS_FAILED_TOTAL = _REGISTRY.register_counter(
    "neuroaegis_jobs_failed_total",
    "Total prediction jobs marked as failed",
    label_keys=("dataset", "execution_mode", "error_category"),
)

JOBS_REAPED_TOTAL = _REGISTRY.register_counter(
    "neuroaegis_jobs_reaped_total",
    "Total stale prediction jobs recovered by the reaper",
    label_keys=(),
)

QUEUE_DISPATCH_FAILURES_TOTAL = _REGISTRY.register_counter(
    "neuroaegis_queue_dispatch_failures_total",
    "Total distributed queue dispatch failures",
    label_keys=("dataset",),
)

WORKER_CLAIM_FAILURES_TOTAL = _REGISTRY.register_counter(
    "neuroaegis_worker_claim_failures_total",
    "Total distributed worker claim collisions or failures",
    label_keys=(),
)

JOBS_IN_PROGRESS = _REGISTRY.register_gauge(
    "neuroaegis_jobs_in_progress",
    "Number of prediction jobs currently being executed in this process",
    label_keys=("execution_mode",),
)

QUEUE_WAIT_SECONDS = _REGISTRY.register_histogram(
    "neuroaegis_queue_wait_seconds",
    "Time spent by job waiting in queue before worker claim in seconds",
    label_keys=("dataset",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

JOB_EXECUTION_SECONDS = _REGISTRY.register_histogram(
    "neuroaegis_job_execution_seconds",
    "Total prediction pipeline execution latency in seconds",
    label_keys=("dataset", "execution_mode", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0),
)

PIPELINE_STAGE_SECONDS = _REGISTRY.register_histogram(
    "neuroaegis_pipeline_stage_seconds",
    "Latency of individual prediction pipeline stages in seconds",
    label_keys=("stage", "dataset", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def safe_telemetry_op(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Safely execute a telemetry operation.
    Guarantees that telemetry failures will never raise to the caller or break business logic.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        # Telemetry must be strictly best-effort and non-throwing
        return None


@contextmanager
def track_stage_latency(stage: str, dataset: str = "unknown"):
    """
    Context manager tracking pipeline stage latency in seconds.
    Observes neuroaegis_pipeline_stage_seconds with status='success' or 'failed'.
    Re-raises original exceptions while suppressing any telemetry failures.
    """
    start_time = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        duration = max(0.0, time.perf_counter() - start_time)
        safe_telemetry_op(
            PIPELINE_STAGE_SECONDS.observe,
            duration,
            labels={"stage": stage, "dataset": dataset, "status": status},
        )
