from __future__ import annotations
import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.db.models import RefreshToken, User

logger = logging.getLogger("neuroaegis.auth")
MAX_PASSWORD_BYTES = 72
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False,
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time verification of password using standard bcrypt."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:MAX_PASSWORD_BYTES],
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Generate bcrypt hash with random salt. Enforces max 72 bytes."""
    pwd_bytes = password.encode("utf-8")
    if len(pwd_bytes) > MAX_PASSWORD_BYTES:
        raise ValueError("Password cannot exceed 72 bytes")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def create_access_token(
    user: User,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Issue short-lived JWT access token with minimal, safe claims.
    No PHI, patient info, EEG data, or secrets are included.
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": str(user.id),
        "username": str(user.username),
        "tenant_id": str(user.tenant_id),
        "role": str(user.role),
        "token_version": int(user.token_version),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(uuid.uuid4()),
        "typ": "access",
    }
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(user: User, db: Session) -> tuple[str, RefreshToken]:
    """
    Generate opaque, high-entropy refresh token.
    Stores only SHA-256 hash in database.
    Returns (raw_opaque_token, refresh_token_model).
    """
    raw_token = secrets.token_urlsafe(64)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    session_record = RefreshToken(
        id=str(uuid.uuid4()),
        token_hash=token_hash,
        user_id=user.id,
        tenant_id=user.tenant_id,
        expires_at=expires_at,
        created_at=datetime.now(timezone.utc),
        revoked_at=None,
        replaced_by=None,
    )
    db.add(session_record)
    db.commit()
    db.refresh(session_record)
    return raw_token, session_record


def verify_and_rotate_refresh_token(
    raw_token: str,
    db: Session,
) -> tuple[User, str, RefreshToken]:
    """
    Verify incoming opaque refresh token and rotate it.
    Detects replay attacks: if an already revoked/replaced token is presented,
    revokes ALL sessions for that user.
    Returns (user, new_raw_token, new_refresh_token_record).
    """
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    incoming_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    session_record = db.query(RefreshToken).filter(RefreshToken.token_hash == incoming_hash).first()

    if not session_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    # Replay attack detection: token was already revoked or replaced
    if session_record.revoked_at is not None or session_record.replaced_by is not None:
        logger.warning(
            f"Security alert: Replay detected on refresh token {session_record.id} for user {session_record.user_id}. Revoking all sessions."
        )
        db.query(RefreshToken).filter(
            RefreshToken.user_id == session_record.user_id,
            RefreshToken.revoked_at.is_(None),
        ).update({"revoked_at": datetime.now(timezone.utc)})
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or reused",
        )

    # Expiration check
    record_expires = session_record.expires_at
    if record_expires.tzinfo is None:
        record_expires = record_expires.replace(tzinfo=timezone.utc)
    if record_expires < datetime.now(timezone.utc):
        session_record.revoked_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )

    user = db.query(User).filter(User.id == session_record.user_id).first()
    if not user or not user.is_active:
        session_record.revoked_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account inactive or not found",
        )

    # Rotate token: revoke old token and create new
    new_raw_token = secrets.token_urlsafe(64)
    new_hash = hashlib.sha256(new_raw_token.encode("utf-8")).hexdigest()
    new_expires = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    new_session = RefreshToken(
        id=str(uuid.uuid4()),
        token_hash=new_hash,
        user_id=user.id,
        tenant_id=user.tenant_id,
        expires_at=new_expires,
        created_at=datetime.now(timezone.utc),
        revoked_at=None,
        replaced_by=None,
    )
    session_record.revoked_at = datetime.now(timezone.utc)
    session_record.replaced_by = new_session.id
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return user, new_raw_token, new_session


def revoke_refresh_token(raw_token: str, db: Session) -> bool:
    """Revoke a single refresh token session."""
    if not raw_token:
        return False
    incoming_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    session_record = db.query(RefreshToken).filter(RefreshToken.token_hash == incoming_hash).first()
    if session_record and session_record.revoked_at is None:
        session_record.revoked_at = datetime.now(timezone.utc)
        db.commit()
        return True
    return False


def revoke_all_user_sessions(user: User, db: Session) -> None:
    """Invalidates all sessions by bumping token_version and revoking all refresh tokens."""
    user.token_version += 1
    db.query(RefreshToken).filter(
        RefreshToken.user_id == user.id,
        RefreshToken.revoked_at.is_(None),
    ).update({"revoked_at": datetime.now(timezone.utc)})
    db.commit()


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: str,
) -> None:
    """Attach secure HttpOnly cookies and CSRF cookie to response."""
    # Access Token (HttpOnly, SameSite=Strict)
    response.set_cookie(
        key=settings.ACCESS_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )
    # Refresh Token (HttpOnly, SameSite=Strict, restricted path)
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        domain=settings.COOKIE_DOMAIN,
        path=settings.REFRESH_COOKIE_PATH,
    )
    # CSRF Token (Readable by JS, SameSite=Strict)
    response.set_cookie(
        key=settings.CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )


def clear_auth_cookies(response: Response) -> None:
    """Clear all authentication and CSRF cookies."""
    response.delete_cookie(
        key=settings.ACCESS_COOKIE_NAME,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        domain=settings.COOKIE_DOMAIN,
        path=settings.REFRESH_COOKIE_PATH,
    )
    response.delete_cookie(
        key=settings.CSRF_COOKIE_NAME,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )


async def get_current_user(
    request: Request,
    token_from_header: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Authoritative authentication dependency:
    1. Extracts token from HttpOnly cookie (primary) or Bearer header (fallback).
    2. Validates signature and expiration.
    3. Validates required claims (sub, tenant_id, role, token_version).
    4. Loads User and verifies user.is_active.
    5. Verifies user.token_version matches token claim.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = request.cookies.get(settings.ACCESS_COOKIE_NAME)
    if not token and token_from_header:
        token = token_from_header

    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": True},
        )
    except JWTError:
        raise credentials_exception

    user_id: Optional[str] = payload.get("sub")
    tenant_id: Optional[str] = payload.get("tenant_id")
    role: Optional[str] = payload.get("role")
    token_version: Optional[int] = payload.get("token_version")

    if not user_id or not tenant_id or not role or token_version is None:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.token_version != token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def verify_csrf_token(request: Request) -> None:
    """
    Validate CSRF for state-changing requests when authenticated via cookies.
    GET/HEAD/OPTIONS are exempt.
    Non-cookie requests (e.g. pure Bearer token without cookies or public endpoints) are exempt.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return

    has_access_cookie = settings.ACCESS_COOKIE_NAME in request.cookies
    has_refresh_cookie = settings.REFRESH_COOKIE_NAME in request.cookies

    if not (has_access_cookie or has_refresh_cookie):
        return

    if request.url.path.endswith("/auth/login"):
        return

    csrf_cookie = request.cookies.get(settings.CSRF_COOKIE_NAME)
    csrf_header = (
        request.headers.get("x-csrf-token")
        or request.headers.get("X-CSRF-Token")
        or request.headers.get("x-csrftoken")
    )

    if not csrf_cookie or not csrf_header or not hmac.compare_digest(csrf_cookie, csrf_header):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing or invalid",
        )


def require_role(required_role: str):
    def role_checker(current_user: User = Depends(get_current_user)):
        roles = ["clinician", "researcher", "admin"]
        if current_user.role not in roles:
            raise HTTPException(status_code=403, detail="Invalid role assigned to user")

        if required_role == "admin" and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        if required_role == "researcher" and current_user.role not in ["researcher", "admin"]:
            raise HTTPException(status_code=403, detail="Researcher access required")

        return current_user
    return role_checker
