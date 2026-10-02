from __future__ import annotations
import concurrent.futures
import math
import pytest

from app.core.errors import (
    SAFE_ERROR_GENERIC,
    SAFE_ERROR_INFERENCE,
    SAFE_ERROR_LEASE_EXPIRED,
    SAFE_JOB_ERRORS,
)
from app.core.telemetry import (
    ALLOWED_DATASETS,
    ALLOWED_EXECUTION_MODES,
    ALLOWED_STAGES,
    ALLOWED_STATUSES,
    PROHIBITED_LABEL_KEYS,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    PIPELINE_STAGE_SECONDS,
    get_telemetry_registry,
    safe_telemetry_op,
    track_stage_latency,
    validate_metric_labels,
)


@pytest.fixture(autouse=True)
def reset_telemetry():
    """Ensure every test runs with clean metrics."""
    get_telemetry_registry().clear()
    yield
    get_telemetry_registry().clear()


def test_counter_lifecycle():
    counter = Counter("test_counter", "Test counter description", label_keys=("dataset", "execution_mode"))
    assert counter.get_value(labels={"dataset": "bonn", "execution_mode": "in_process"}) == 0.0

    counter.inc(labels={"dataset": "bonn", "execution_mode": "in_process"})
    assert counter.get_value(labels={"dataset": "bonn", "execution_mode": "in_process"}) == 1.0

    counter.inc(3.5, labels={"dataset": "bonn", "execution_mode": "in_process"})
    assert counter.get_value(labels={"dataset": "bonn", "execution_mode": "in_process"}) == 4.5

    # Distinct labels
    counter.inc(2.0, labels={"dataset": "chbmit", "execution_mode": "distributed"})
    assert counter.get_value(labels={"dataset": "chbmit", "execution_mode": "distributed"}) == 2.0
    assert counter.get_value(labels={"dataset": "bonn", "execution_mode": "in_process"}) == 4.5

    # Negative increment rejected
    with pytest.raises(ValueError, match="non-negative"):
        counter.inc(-1.0, labels={"dataset": "bonn", "execution_mode": "in_process"})


def test_counter_without_labels():
    counter = Counter("test_counter_nolabels", "No labels", label_keys=())
    assert counter.get_value() == 0.0
    counter.inc()
    assert counter.get_value() == 1.0
    counter.inc(5.0)
    assert counter.get_value() == 6.0


def test_gauge_lifecycle():
    gauge = Gauge("test_gauge", "Test gauge description", label_keys=("execution_mode",))
    assert gauge.get_value(labels={"execution_mode": "in_process"}) == 0.0

    gauge.inc(2.0, labels={"execution_mode": "in_process"})
    assert gauge.get_value(labels={"execution_mode": "in_process"}) == 2.0

    gauge.dec(1.0, labels={"execution_mode": "in_process"})
    assert gauge.get_value(labels={"execution_mode": "in_process"}) == 1.0

    # Decrement below zero is clamped to 0.0
    gauge.dec(5.0, labels={"execution_mode": "in_process"})
    assert gauge.get_value(labels={"execution_mode": "in_process"}) == 0.0

    gauge.set(10.0, labels={"execution_mode": "in_process"})
    assert gauge.get_value(labels={"execution_mode": "in_process"}) == 10.0


def test_histogram_lifecycle_and_buckets():
    buckets = (0.1, 0.5, 1.0, 5.0)
    hist = Histogram("test_hist", "Test histogram", label_keys=("dataset",), buckets=buckets)

    # Observe samples
    samples = [0.05, 0.1, 0.4, 0.5, 0.8, 1.2, 7.5]
    for s in samples:
        hist.observe(s, labels={"dataset": "bonn"})

    stats = hist.get_stats(labels={"dataset": "bonn"})
    assert stats["count"] == len(samples)
    assert math.isclose(stats["sum"], sum(samples), rel_tol=1e-5)

    # Verify cumulative bucket counts
    # <= 0.1: 0.05, 0.1 (2)
    assert stats["buckets"][0.1] == 2
    # <= 0.5: 0.05, 0.1, 0.4, 0.5 (4)
    assert stats["buckets"][0.5] == 4
    # <= 1.0: 0.05, 0.1, 0.4, 0.5, 0.8 (5)
    assert stats["buckets"][1.0] == 5
    # <= 5.0: 0.05, 0.1, 0.4, 0.5, 0.8, 1.2 (6)
    assert stats["buckets"][5.0] == 6

    # Verify Prometheus exposition lines
    lines = hist.format_prometheus()
    assert any('test_hist_bucket{dataset="bonn",le="0.1"} 2' in l for l in lines)
    assert any('test_hist_bucket{dataset="bonn",le="0.5"} 4' in l for l in lines)
    assert any('test_hist_bucket{dataset="bonn",le="1"} 5' in l for l in lines)
    assert any('test_hist_bucket{dataset="bonn",le="5"} 6' in l for l in lines)
    assert any('test_hist_bucket{dataset="bonn",le="+Inf"} 7' in l for l in lines)
    assert any('test_hist_count{dataset="bonn"} 7' in l for l in lines)


def test_label_validation_strict_enums():
    # Valid labels pass
    labels = {"dataset": "bonn", "execution_mode": "in_process"}
    validated = validate_metric_labels(("dataset", "execution_mode"), labels)
    assert validated == labels

    # Invalid status
    with pytest.raises(ValueError, match="Invalid status"):
        validate_metric_labels(("status",), {"status": "running"})

    # Invalid execution_mode
    with pytest.raises(ValueError, match="Invalid execution_mode"):
        validate_metric_labels(("execution_mode",), {"execution_mode": "async_task"})

    # Invalid stage
    with pytest.raises(ValueError, match="Invalid stage"):
        validate_metric_labels(("stage",), {"stage": "unregistered_stage"})

    # Invalid dataset
    with pytest.raises(ValueError, match="Invalid dataset"):
        validate_metric_labels(("dataset",), {"dataset": "custom_hospital_eeg"})

    # Invalid error_category
    with pytest.raises(ValueError, match="Invalid error_category"):
        validate_metric_labels(("error_category",), {"error_category": "Traceback (most recent call last): ..."})

    # Valid error_category
    for err in SAFE_JOB_ERRORS:
        val = validate_metric_labels(("error_category",), {"error_category": err})
        assert val["error_category"] == err


def test_prohibited_label_keys_rejected():
    for prohibited in PROHIBITED_LABEL_KEYS:
        with pytest.raises(ValueError, match="Prohibited"):
            validate_metric_labels((prohibited,), {prohibited: "value"})


def test_missing_and_extra_labels():
    # Missing key
    with pytest.raises(ValueError, match="Label mismatch"):
        validate_metric_labels(("dataset", "execution_mode"), {"dataset": "bonn"})

    # Extra key
    with pytest.raises(ValueError, match="Label mismatch"):
        validate_metric_labels(("dataset",), {"dataset": "bonn", "unexpected": "extra"})

    # Metric expecting no labels given labels
    with pytest.raises(ValueError, match="Metric does not accept labels"):
        validate_metric_labels((), {"dataset": "bonn"})

    # Metric expecting labels given None
    with pytest.raises(ValueError, match="requires labels"):
        validate_metric_labels(("dataset",), None)


def test_safe_telemetry_op_never_raises():
    def throwing_fn():
        raise RuntimeError("Telemetry failure inside metric observation")

    # Does not raise, returns None
    result = safe_telemetry_op(throwing_fn)
    assert result is None

    # Normal function returns its return value
    assert safe_telemetry_op(lambda x, y: x + y, 3, 4) == 7


def test_track_stage_latency_success_and_failure():
    # Success observation
    with track_stage_latency("preprocessing", dataset="bonn"):
        _ = 1 + 1

    stats_success = PIPELINE_STAGE_SECONDS.get_stats(
        labels={"stage": "preprocessing", "dataset": "bonn", "status": "success"}
    )
    assert stats_success["count"] == 1
    assert stats_success["sum"] >= 0.0

    # Failure observation: must re-raise original exception while recording status='failed'
    with pytest.raises(ZeroDivisionError):
        with track_stage_latency("feature_extraction", dataset="chbmit"):
            _ = 1 / 0

    stats_failed = PIPELINE_STAGE_SECONDS.get_stats(
        labels={"stage": "feature_extraction", "dataset": "chbmit", "status": "failed"}
    )
    assert stats_failed["count"] == 1
    assert stats_failed["sum"] >= 0.0


def test_prometheus_exposition_escaping():
    reg = MetricsRegistry()
    c = reg.register_counter("test_escape_total", "Test escaping", label_keys=("custom_label",))
    c.inc(labels={"custom_label": 'bonn"val\nxt\\end'})

    output = reg.generate_prometheus_text()
    assert 'test_escape_total{custom_label="bonn\\"val\\nxt\\\\end"} 1' in output


def test_thread_safety():
    counter = Counter("concurrent_counter", "Concurrent test", label_keys=("dataset", "execution_mode"))
    num_threads = 8
    increments_per_thread = 200

    def worker():
        for _ in range(increments_per_thread):
            counter.inc(labels={"dataset": "bonn", "execution_mode": "in_process"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for f in futures:
            f.result()

    assert counter.get_value(labels={"dataset": "bonn", "execution_mode": "in_process"}) == float(
        num_threads * increments_per_thread
    )
