from __future__ import annotations
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.auth import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    revoke_all_user_sessions,
    verify_password,
)
from app.core.config import DEFAULT_DEV_SECRET_KEY, settings
from app.db.database import Base, ensure_schema_compatibility, get_db
from app.db.models import DEFAULT_TENANT_ID, RefreshToken, Tenant, User
from app.main import app

# Shared test database
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)
ensure_schema_compatibility(test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure clean database and dependency override before every test."""
    app.dependency_overrides[get_db] = override_get_db
    db = TestingSessionLocal()
    try:
        db.query(RefreshToken).delete()
        db.query(User).delete()
        db.commit()
    finally:
        db.close()
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_user():
    db = TestingSessionLocal()
    user = User(
        id=str(uuid.uuid4()),
        username="dr_auth_test",
        hashed_password=get_password_hash("CorrectPassword123!"),
        role="clinician",
        tenant_id=DEFAULT_TENANT_ID,
        is_active=True,
        token_version=1,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user


@pytest.fixture
def inactive_user():
    db = TestingSessionLocal()
    user = User(
        id=str(uuid.uuid4()),
        username="dr_inactive",
        hashed_password=get_password_hash("Password123!"),
        role="clinician",
        tenant_id=DEFAULT_TENANT_ID,
        is_active=False,
        token_version=1,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user


# ==============================================================================
# 1. Login Tests
# ==============================================================================

def test_login_valid_credentials(client: TestClient, test_user: User):
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" not in data
    assert "refresh_token" not in data
    assert "token_type" not in data
    assert "csrf_token" in data
    assert data["user"]["username"] == "dr_auth_test"
    assert data["user"]["id"] == test_user.id
    assert data["user"]["tenant_id"] == DEFAULT_TENANT_ID
    assert "hashed_password" not in data["user"]
    assert "password" not in data["user"]
    assert "hashed_password" not in data
    assert "password" not in data

    # Verify cookies attached
    cookies = resp.cookies
    assert settings.ACCESS_COOKIE_NAME in cookies
    assert settings.REFRESH_COOKIE_NAME in cookies
    assert settings.CSRF_COOKIE_NAME in cookies

    # Verify access cookie is HttpOnly, SameSite=Strict, Path=/
    raw_cookies = resp.headers.get_list("set-cookie")
    access_cookie_str = next(c for c in raw_cookies if settings.ACCESS_COOKIE_NAME in c)
    assert "HttpOnly" in access_cookie_str
    assert "samesite=strict" in access_cookie_str.lower()
    assert "Path=/" in access_cookie_str


def test_login_invalid_password(client: TestClient, test_user: User):
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "WrongPassword!"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Incorrect username or password"
    # Ensure no auth cookies set
    assert settings.ACCESS_COOKIE_NAME not in resp.cookies


def test_login_nonexistent_user(client: TestClient):
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "nonexistent_dr", "password": "AnyPassword123!"},
    )
    assert resp.status_code == 401
    # Generic error preventing account enumeration
    assert resp.json()["detail"] == "Incorrect username or password"


def test_login_inactive_user(client: TestClient, inactive_user: User):
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_inactive", "password": "Password123!"},
    )
    assert resp.status_code == 401
    # Generic error preventing account enumeration
    assert resp.json()["detail"] == "Incorrect username or password"


def test_login_malformed_input(client: TestClient):
    # Missing password
    resp = client.post("/api/v1/auth/login", json={"username": "dr_auth"})
    assert resp.status_code == 422

    # Empty payload
    resp2 = client.post("/api/v1/auth/login", json={})
    assert resp2.status_code == 422

    # Invalid non-json body
    resp3 = client.post(
        "/api/v1/auth/login",
        content="not a json",
        headers={"Content-Type": "application/json"},
    )
    assert resp3.status_code == 422


def test_login_form_encoded(client: TestClient, test_user: User):
    """Ensure form-encoded requests (e.g. Swagger) work alongside JSON."""
    resp = client.post(
        "/api/v1/auth/login",
        data={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" not in data
    assert "refresh_token" not in data
    assert settings.ACCESS_COOKIE_NAME in resp.cookies


def test_login_security_boundary_contract(client: TestClient, test_user: User):
    """
    Prompt 9.2.1 Security Contract:
    1. successful login sets access cookie
    2. access cookie is HttpOnly
    3. login JSON does NOT contain access_token
    4. login JSON does NOT contain refresh_token
    5. login JSON does not contain password/hash
    6. /me works using the authentication cookie
    7. existing Bearer authentication tests continue to pass
    """
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # 1. successful login sets access cookie
    assert settings.ACCESS_COOKIE_NAME in resp.cookies
    access_cookie_val = resp.cookies[settings.ACCESS_COOKIE_NAME]
    assert access_cookie_val

    # 2. access cookie is HttpOnly
    raw_cookies = resp.headers.get_list("set-cookie")
    access_cookie_header = next(c for c in raw_cookies if settings.ACCESS_COOKIE_NAME in c)
    assert "HttpOnly" in access_cookie_header

    # 3. login JSON does NOT contain access_token
    assert "access_token" not in data

    # 4. login JSON does NOT contain refresh_token
    assert "refresh_token" not in data

    # 5. login JSON does not contain password/hash
    assert "password" not in data
    assert "hashed_password" not in data
    if "user" in data:
        assert "password" not in data["user"]
        assert "hashed_password" not in data["user"]

    # 6. /me works using the authentication cookie
    me_cookie_resp = client.get("/api/v1/auth/me")
    assert me_cookie_resp.status_code == 200
    assert me_cookie_resp.json()["id"] == test_user.id

    # 7. existing Bearer authentication tests continue to pass
    # Using a client without cookies and explicit Bearer header
    token = create_access_token(test_user)
    bearer_client = TestClient(app)
    me_bearer_resp = bearer_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_bearer_resp.status_code == 200
    assert me_bearer_resp.json()["id"] == test_user.id


# ==============================================================================
# 2. Access Token Tests
# ==============================================================================

def test_access_token_claims(test_user: User):
    token = create_access_token(test_user)
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    assert payload["sub"] == test_user.id
    assert payload["username"] == test_user.username
    assert payload["tenant_id"] == test_user.tenant_id
    assert payload["role"] == test_user.role
    assert payload["token_version"] == test_user.token_version
    assert payload["typ"] == "access"
    assert "iat" in payload
    assert "exp" in payload
    assert "jti" in payload
    assert payload["exp"] > payload["iat"]


def test_access_token_expired(client: TestClient, test_user: User):
    expired_token = create_access_token(test_user, expires_delta=timedelta(seconds=-10))
    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert resp.status_code == 401


def test_access_token_invalid_signature(client: TestClient, test_user: User):
    bad_key_token = jwt.encode(
        {"sub": test_user.id, "tenant_id": test_user.tenant_id, "role": test_user.role, "token_version": 1},
        "wrong_secret_key_12345678901234567890",
        algorithm="HS256",
    )
    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {bad_key_token}"},
    )
    assert resp.status_code == 401


def test_access_token_missing_required_claims(client: TestClient):
    # Missing tenant_id & token_version
    incomplete_token = jwt.encode(
        {"sub": str(uuid.uuid4()), "role": "clinician", "exp": 9999999999},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {incomplete_token}"},
    )
    assert resp.status_code == 401


def test_access_token_stale_token_version(client: TestClient, test_user: User):
    token = create_access_token(test_user)

    # Bump token_version on user
    db = TestingSessionLocal()
    u = db.query(User).filter_by(id=test_user.id).one()
    u.token_version += 1
    db.commit()
    db.close()

    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    assert "revoked" in resp.json()["detail"].lower()


def test_access_token_inactive_user(client: TestClient, test_user: User):
    token = create_access_token(test_user)

    # Deactivate user
    db = TestingSessionLocal()
    u = db.query(User).filter_by(id=test_user.id).one()
    u.is_active = False
    db.commit()
    db.close()

    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    assert "inactive" in resp.json()["detail"].lower()


# ==============================================================================
# 3. Cookie Tests
# ==============================================================================

def test_cookie_flags_on_login(client: TestClient, test_user: User):
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    assert resp.status_code == 200

    raw_cookies = resp.headers.get_list("set-cookie")
    access_cookie_str = next(c for c in raw_cookies if settings.ACCESS_COOKIE_NAME in c)
    refresh_cookie_str = next(c for c in raw_cookies if settings.REFRESH_COOKIE_NAME in c)
    csrf_cookie_str = next(c for c in raw_cookies if settings.CSRF_COOKIE_NAME in c)

    # Access Token: HttpOnly, SameSite=Strict
    assert "HttpOnly" in access_cookie_str
    assert "samesite=strict" in access_cookie_str.lower()
    assert "Path=/" in access_cookie_str

    # Refresh Token: HttpOnly, SameSite=Strict, Path=/api/v1/auth
    assert "HttpOnly" in refresh_cookie_str
    assert "samesite=strict" in refresh_cookie_str.lower()
    assert f"Path={settings.REFRESH_COOKIE_PATH}" in refresh_cookie_str

    # CSRF Token: NOT HttpOnly (frontend needs to read it), SameSite=Strict
    assert "HttpOnly" not in csrf_cookie_str
    assert "samesite=strict" in csrf_cookie_str.lower()


def test_cookies_cleared_on_logout(client: TestClient, test_user: User):
    # First login to get cookies
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    csrf_token = login_resp.json()["csrf_token"]

    # Logout with CSRF token
    logout_resp = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert logout_resp.status_code == 200

    # Ensure cookies deleted (max-age=0 or empty)
    raw_cookies = logout_resp.headers.get_list("set-cookie")
    for cookie_header in raw_cookies:
        assert 'max-age=0' in cookie_header.lower() or 'expires=thu, 01 jan 1970' in cookie_header.lower()


# ==============================================================================
# 4. Refresh Token Lifecycle & Replay Detection Tests
# ==============================================================================

def test_refresh_lifecycle_and_rotation(client: TestClient, test_user: User):
    # 1. Login
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    old_access = login_resp.cookies.get(settings.ACCESS_COOKIE_NAME)
    csrf_token = login_resp.json()["csrf_token"]
    old_refresh = login_resp.cookies.get(settings.REFRESH_COOKIE_NAME)

    # 2. Refresh
    refresh_resp = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert refresh_resp.status_code == 200
    new_access = refresh_resp.json()["access_token"]
    new_csrf = refresh_resp.json()["csrf_token"]
    new_refresh = refresh_resp.cookies.get(settings.REFRESH_COOKIE_NAME)

    assert new_access != old_access
    assert new_refresh != old_refresh
    assert new_csrf is not None

    # 3. New access token works
    me_resp = client.get("/api/v1/auth/me")
    assert me_resp.status_code == 200


def test_refresh_token_replay_detection(client: TestClient, test_user: User):
    """
    If a previously rotated refresh token is presented again (replay attack),
    ALL sessions for that user must be immediately revoked.
    """
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    csrf_token = login_resp.json()["csrf_token"]
    initial_refresh = login_resp.cookies.get(settings.REFRESH_COOKIE_NAME)

    # 1. Legitimate refresh rotates token
    refresh_resp1 = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert refresh_resp1.status_code == 200
    new_csrf = refresh_resp1.json()["csrf_token"]

    # 2. Attacker replays old initial_refresh token with current CSRF
    client.cookies.set(settings.REFRESH_COOKIE_NAME, initial_refresh, path=settings.REFRESH_COOKIE_PATH)
    replay_resp = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": new_csrf},
    )
    assert replay_resp.status_code == 401
    assert "revoked or reused" in replay_resp.json()["detail"].lower()

    # 3. Verify ALL sessions for user are now revoked in DB
    db = TestingSessionLocal()
    active_tokens = db.query(RefreshToken).filter(
        RefreshToken.user_id == test_user.id,
        RefreshToken.revoked_at.is_(None),
    ).count()
    db.close()
    assert active_tokens == 0


def test_refresh_expired_token(client: TestClient, test_user: User):
    db = TestingSessionLocal()
    raw_token, session_record = create_refresh_token(test_user, db)
    # Set expiration in the past
    session_record.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    db.commit()
    db.close()

    client.cookies.set(settings.REFRESH_COOKIE_NAME, raw_token, path=settings.REFRESH_COOKIE_PATH)
    client.cookies.set(settings.CSRF_COOKIE_NAME, "test_csrf", path="/")

    resp = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": "test_csrf"},
    )
    assert resp.status_code == 401
    assert "expired" in resp.json()["detail"].lower()


def test_refresh_invalid_token(client: TestClient):
    client.cookies.set(settings.REFRESH_COOKIE_NAME, "completely_bogus_token", path=settings.REFRESH_COOKIE_PATH)
    client.cookies.set(settings.CSRF_COOKIE_NAME, "test_csrf", path="/")

    resp = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": "test_csrf"},
    )
    assert resp.status_code == 401
    assert "invalid" in resp.json()["detail"].lower()


# ==============================================================================
# 5. Logout and Logout-All Tests
# ==============================================================================

def test_logout_revokes_session(client: TestClient, test_user: User):
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    csrf_token = login_resp.json()["csrf_token"]
    refresh_token = login_resp.cookies.get(settings.REFRESH_COOKIE_NAME)

    # Logout
    logout_resp = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert logout_resp.status_code == 200

    # Ensure refresh token is revoked in DB
    db = TestingSessionLocal()
    import hashlib
    token_hash = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()
    rec = db.query(RefreshToken).filter_by(token_hash=token_hash).one()
    assert rec.revoked_at is not None
    db.close()

    # /me should now fail as unauthenticated
    me_resp = client.get("/api/v1/auth/me")
    assert me_resp.status_code == 401


def test_logout_safe_when_already_logged_out(client: TestClient):
    resp = client.post("/api/v1/auth/logout")
    assert resp.status_code == 200
    assert resp.json()["detail"] == "Logged out successfully"


def test_logout_all_invalidates_all_tokens(client: TestClient, test_user: User):
    # Login session 1
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    access_token = login_resp.cookies.get(settings.ACCESS_COOKIE_NAME)
    csrf_token = login_resp.json()["csrf_token"]

    # Logout-all
    logout_all_resp = client.post(
        "/api/v1/auth/logout-all",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert logout_all_resp.status_code == 200

    # 1. Existing JWT access token should now be rejected due to bumped token_version
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_resp.status_code == 401

    # 2. All refresh tokens should be marked revoked
    db = TestingSessionLocal()
    active_count = db.query(RefreshToken).filter(
        RefreshToken.user_id == test_user.id,
        RefreshToken.revoked_at.is_(None),
    ).count()
    u = db.query(User).filter_by(id=test_user.id).one()
    assert u.token_version > 1
    db.close()
    assert active_count == 0


# ==============================================================================
# 6. CSRF Protection Tests
# ==============================================================================

def test_csrf_missing_token_on_cookie_post(client: TestClient, test_user: User):
    # Login establishes cookies
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    assert login_resp.status_code == 200

    # Attempt POST /api/v1/auth/logout without X-CSRF-Token header
    resp = client.post("/api/v1/auth/logout")
    assert resp.status_code == 403
    assert "csrf" in resp.json()["detail"].lower()


def test_csrf_invalid_token_on_cookie_post(client: TestClient, test_user: User):
    client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )

    # Send bogus CSRF header
    resp = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": "bogus_csrf_token"},
    )
    assert resp.status_code == 403
    assert "csrf" in resp.json()["detail"].lower()


def test_csrf_valid_token_accepted(client: TestClient, test_user: User):
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    csrf_token = login_resp.json()["csrf_token"]

    # Send matching CSRF header
    resp = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 200


def test_csrf_safe_get_exempt(client: TestClient, test_user: User):
    client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    # GET /me does not require CSRF header
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 200


def test_csrf_unauthenticated_requests_exempt(client: TestClient):
    # POST /login does not require CSRF
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "nobody", "password": "any"},
    )
    # Fails with 401 authentication error, NOT 403 CSRF
    assert resp.status_code == 401


# ==============================================================================
# 7. /me Endpoint Tests
# ==============================================================================

def test_me_unauthenticated(client: TestClient):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_me_authenticated_cookie(client: TestClient, test_user: User):
    client.post(
        "/api/v1/auth/login",
        json={"username": "dr_auth_test", "password": "CorrectPassword123!"},
    )
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == test_user.id
    assert data["username"] == "dr_auth_test"
    assert data["tenant_id"] == DEFAULT_TENANT_ID
    assert data["role"] == "clinician"
    assert data["is_active"] is True
    # Ensure sensitive credentials are absent
    assert "hashed_password" not in data
    assert "password" not in data


def test_me_authenticated_bearer(client: TestClient, test_user: User):
    token = create_access_token(test_user)
    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "dr_auth_test"


# ==============================================================================
# 8. Security Invariant Tests
# ==============================================================================

def test_no_phi_in_jwt(test_user: User):
    token = create_access_token(test_user)
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    forbidden_keys = {
        "patient", "patient_id", "medical_history", "eeg", "eeg_data",
        "channels", "prediction", "shap", "diagnosis", "features"
    }
    assert forbidden_keys.isdisjoint(set(payload.keys()))


def test_password_max_length_enforced():
    with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
        get_password_hash("A" * 73)


def test_password_hash_verification():
    pw = "SuperSecurePassword987!"
    h = get_password_hash(pw)
    assert h.startswith("$2b$")
    assert verify_password(pw, h)
    assert not verify_password("WrongPassword", h)
    assert not verify_password("", h)
