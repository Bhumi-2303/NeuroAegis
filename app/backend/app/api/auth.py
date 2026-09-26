from __future__ import annotations
import secrets
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.db.models import User
from app.core.auth import (
    clear_auth_cookies,
    create_access_token,
    create_refresh_token,
    get_current_user,
    revoke_all_user_sessions,
    revoke_refresh_token,
    set_auth_cookies,
    verify_and_rotate_refresh_token,
    verify_csrf_token,
    verify_password,
)

router = APIRouter()


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Authenticate user credentials, establish server-side refresh session,
    issue short-lived access JWT, and attach secure HttpOnly cookies.
    """
    username: str | None = None
    password: str | None = None

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                username = body.get("username")
                password = body.get("password")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Malformed JSON body",
            )
    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        try:
            form = await request.form()
            username = form.get("username")
            password = form.get("password")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Malformed form data",
            )
    else:
        try:
            body = await request.json()
            if isinstance(body, dict):
                username = body.get("username")
                password = body.get("password")
        except Exception:
            pass

    if not username or not password or not isinstance(username, str) or not isinstance(password, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username and password are required",
        )

    # Generic error message to prevent account enumeration
    generic_auth_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Dummy verification to prevent timing discrepancy
        verify_password("dummy", "$2b$12$DUMMY_SALT_DUMMY_SALT_DUMMY_SALT_DUMMY_SALT_DUMMY")
        raise generic_auth_error

    if not verify_password(password, user.hashed_password):
        raise generic_auth_error

    if not user.is_active:
        raise generic_auth_error

    # Issue credentials
    access_token = create_access_token(user)
    raw_refresh_token, _ = create_refresh_token(user, db)
    csrf_token = secrets.token_urlsafe(32)

    set_auth_cookies(response, access_token, raw_refresh_token, csrf_token)

    return {
        "csrf_token": csrf_token,
        "user": {
            "id": user.id,
            "username": user.username,
            "tenant_id": user.tenant_id,
            "role": user.role,
        },
    }


@router.get("/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Return authenticated caller safe identity metadata.
    Never exposes passwords, hashes, or internal secrets.
    """
    return {
        "id": current_user.id,
        "username": current_user.username,
        "tenant_id": current_user.tenant_id,
        "role": current_user.role,
        "is_active": current_user.is_active,
    }


@router.post("/refresh")
async def refresh_session(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Rotate refresh token, enforce replay protection, and issue new access token.
    """
    verify_csrf_token(request)

    raw_token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if not raw_token:
        try:
            body = await request.json()
            if isinstance(body, dict):
                raw_token = body.get("refresh_token")
        except Exception:
            pass

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    user, new_raw_token, _ = verify_and_rotate_refresh_token(raw_token, db)
    new_access_token = create_access_token(user)
    new_csrf_token = secrets.token_urlsafe(32)

    set_auth_cookies(response, new_access_token, new_raw_token, new_csrf_token)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "csrf_token": new_csrf_token,
    }


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Revoke current refresh session and clear authentication cookies.
    Safe to call even if already logged out.
    """
    verify_csrf_token(request)

    raw_token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if raw_token:
        revoke_refresh_token(raw_token, db)

    clear_auth_cookies(response)
    return {"detail": "Logged out successfully"}


@router.post("/logout-all")
async def logout_all(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Revoke all active refresh sessions and increment token_version to invalidate existing JWTs.
    """
    verify_csrf_token(request)

    revoke_all_user_sessions(current_user, db)
    clear_auth_cookies(response)
    return {"detail": "All sessions revoked successfully"}
