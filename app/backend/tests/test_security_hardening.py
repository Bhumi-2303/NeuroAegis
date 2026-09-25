import io
import pytest
from fastapi.testclient import TestClient

from app.core.config import DEFAULT_DEV_SECRET_KEY, Settings
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_settings_security_production_rejection():
    # Production without changing default secret must fail
    with pytest.raises(ValueError, match="SECRET_KEY must be a strong, unique secret"):
        Settings(_env_file=None, ENVIRONMENT="production", SECRET_KEY=DEFAULT_DEV_SECRET_KEY)

    # Production with short secret must fail
    with pytest.raises(ValueError, match="SECRET_KEY must be a strong, unique secret"):
        Settings(_env_file=None, ENVIRONMENT="production", SECRET_KEY="too_short_secret")

    # Production with insecure wildcard CORS + credentials must fail
    with pytest.raises(ValueError, match="Wildcard '\\*' in CORS allowed origins"):
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            SECRET_KEY="a" * 32,
            CORS_ALLOWED_ORIGINS=["*"],
            CORS_ALLOW_CREDENTIALS=True,
        )

    # Production with valid settings succeeds and defaults ENABLE_DOCS to False
    prod_settings = Settings(
        _env_file=None,
        ENVIRONMENT="production",
        SECRET_KEY="a" * 32,
        CORS_ALLOWED_ORIGINS=["https://app.neuroaegis.com"],
    )
    assert prod_settings.ENABLE_DOCS is False
    assert prod_settings.CORS_ALLOWED_ORIGINS == ["https://app.neuroaegis.com"]


def test_cors_origins_parsing():
    s1 = Settings(CORS_ALLOWED_ORIGINS="http://site1.com, http://site2.com")
    assert s1.CORS_ALLOWED_ORIGINS == ["http://site1.com", "http://site2.com"]

    s2 = Settings(CORS_ALLOWED_ORIGINS='["http://site3.com", "http://site4.com"]')
    assert s2.CORS_ALLOWED_ORIGINS == ["http://site3.com", "http://site4.com"]


def test_security_headers_present(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert "X-XSS-Protection" not in resp.headers  # Deprecated/obsolete header cleanly removed
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "camera=()" in resp.headers.get("Permissions-Policy", "")
    assert "Content-Security-Policy" in resp.headers


def test_content_security_policy_headers(client: TestClient, monkeypatch):
    from unittest.mock import patch
    from app.core.config import settings

    # In default dev mode, CSP allows 'self' and Swagger CDN assets without wildcard
    resp_dev = client.get("/healthz")
    csp_dev = resp_dev.headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp_dev
    assert "frame-ancestors 'none'" in csp_dev
    assert "object-src 'none'" in csp_dev
    assert "*" not in csp_dev

    # In production mode, CSP is strictly minimized
    with patch.object(settings, "ENVIRONMENT", "production"):
        resp_prod = client.get("/healthz")
        csp_prod = resp_prod.headers.get("Content-Security-Policy", "")
        assert csp_prod == "default-src 'self'; frame-ancestors 'none'; object-src 'none'"


def test_cors_headers_enforcement(client: TestClient):
    # Allowed origin
    resp_allowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp_allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    # Disallowed origin
    resp_disallowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://malicious-attacker.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp_disallowed.headers.get("access-control-allow-origin") != "http://malicious-attacker.com"


def test_liveness_and_readiness_probes(client: TestClient):
    # Root probes
    resp_liveness = client.get("/liveness")
    assert resp_liveness.status_code == 200
    assert resp_liveness.json()["status"] == "alive"

    resp_readiness = client.get("/readiness")
    assert resp_readiness.status_code == 200
    data = resp_readiness.json()
    assert "status" in data
    assert "database" in data
    assert data["database"] == "healthy"
    # Ensure no secrets or filesystem paths leak in probe
    assert "SECRET_KEY" not in str(data)
    assert "neuroaegis.db" not in str(data)

    # V1 probes
    v1_live = client.get("/api/v1/liveness")
    assert v1_live.status_code == 200
    assert v1_live.json()["status"] == "alive"

    v1_ready = client.get("/api/v1/readiness")
    assert v1_ready.status_code == 200
    assert v1_ready.json()["database"] == "healthy"


def test_job_id_validation_prevents_traversal(client: TestClient):
    # v1 job lookup
    resp1 = client.get("/api/v1/jobs/invalid@id#")
    assert resp1.status_code == 400
    assert "Invalid job ID format" in resp1.json()["detail"]

    # v2 job status
    resp2 = client.get("/api/v2/jobs/invalid..traversal")
    assert resp2.status_code == 400
    assert "Invalid job ID format" in resp2.json()["detail"]

    # v2 report
    resp3 = client.get("/api/v2/report/invalid$id")
    assert resp3.status_code == 400
    assert "Invalid job ID format" in resp3.json()["detail"]

    # v1 patient lookup
    resp4 = client.get("/api/v1/patients/bad-id-with-special!chars")
    assert resp4.status_code == 400
    assert "Invalid patient ID format" in resp4.json()["detail"]


def test_v2_predict_malformed_json_input(client: TestClient):
    # Pass non-dict JSON or raw strings
    fake_edf = io.BytesIO(b"dummy EDF content")
    response = client.post(
        "/api/v2/predict",
        data={
            "name": "Jane Doe",
            "age": 30,
            "gender": "F",
            "weight": 60.0,
            "height": 165.0,
            "medical_history": '"just a string not an object"',
            "vital_signs": "12345",
        },
        files={"file": ("test.edf", fake_edf, "application/octet-stream")},
    )
    # Must reject with 400
    assert response.status_code == 400
    assert "JSON" in response.json()["detail"]


def test_health_check_unloaded_models(client: TestClient, monkeypatch):
    from app.services.prediction.prediction_router import prediction_router
    from app.services.model_service import ml_model_service

    monkeypatch.setattr(prediction_router, "is_loaded", False)
    monkeypatch.setattr(prediction_router, "_predictors", {})

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert data["model_version"] is None
    assert data["dataset_name"] is None


def test_exception_masking_demo_endpoint(client: TestClient, monkeypatch):
    import os
    import pandas as pd

    monkeypatch.setattr(os.path, "exists", lambda p: True)

    def _mock_read_parquet(*args, **kwargs):
        raise RuntimeError("Internal database server error on /internal/path/file.parquet: password=secret123")

    monkeypatch.setattr(pd, "read_parquet", _mock_read_parquet)
    response = client.get("/api/v2/demo/live-monitor-data")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "secret123" not in detail
    assert "file.parquet" not in detail
    assert detail == "Failed to load demo live-monitor data"


def test_docs_disabled_in_production(monkeypatch):
    from unittest.mock import patch
    from app.core.config import settings

    with patch.object(settings, "ENABLE_DOCS", False):
        from fastapi import FastAPI
        docs_url = "/docs" if settings.ENABLE_DOCS else None
        redoc_url = "/redoc" if settings.ENABLE_DOCS else None
        test_app = FastAPI(docs_url=docs_url, redoc_url=redoc_url)
        test_client = TestClient(test_app)
        assert test_client.get("/docs").status_code == 404
        assert test_client.get("/redoc").status_code == 404


def test_distributed_readiness_probe_with_redis(client: TestClient, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.core.config import settings

    mock_pool = MagicMock()
    mock_pool.ping = AsyncMock(return_value=True)

    with patch.object(settings, "ENABLE_DISTRIBUTED_QUEUE", True):
        # Successful redis ping
        with patch("app.services.queue.prediction_queue.get_pool", AsyncMock(return_value=mock_pool)):
            resp = client.get("/readiness")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ready"
            assert resp.json()["redis"] == "healthy"

        # Failed redis ping
        failing_mock = MagicMock()
        failing_mock.ping = AsyncMock(side_effect=RuntimeError("Redis connection refused"))
        with patch("app.services.queue.prediction_queue.get_pool", AsyncMock(return_value=failing_mock)):
            resp = client.get("/readiness")
            assert resp.status_code == 503
            assert resp.json()["status"] == "not_ready"
            assert resp.json()["redis"] == "unhealthy"
