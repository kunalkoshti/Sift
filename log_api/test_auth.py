import asyncio
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from log_api.auth import AuthConfig, SupabaseJWTVerifier, create_admin_dependency


def test_legacy_supabase_hs256_token_is_verified(monkeypatch):
    issuer = "https://example.supabase.co/auth/v1"
    secret = "local-test-secret-with-at-least-32-bytes"
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SUPABASE_JWT_ISSUER", issuer)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)

    token = jwt.encode(
        {
            "sub": "user-123",
            "aud": "authenticated",
            "iss": issuer,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    verifier = SupabaseJWTVerifier(AuthConfig.from_env())

    claims = asyncio.run(
        verifier(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    )

    assert claims is not None
    assert claims["sub"] == "user-123"


def test_admin_dependency_allows_configured_email(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "owner@example.com")
    dependency = create_admin_dependency(lambda: None)

    claims = asyncio.run(dependency(claims={"email": "owner@example.com"}))

    assert claims["email"] == "owner@example.com"


def test_admin_dependency_rejects_non_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "owner@example.com")
    dependency = create_admin_dependency(lambda: None)

    async def exercise():
        try:
            await dependency(claims={"email": "other@example.com"})
        except HTTPException as exc:
            return exc
        raise AssertionError("expected admin authorization to fail")

    error = asyncio.run(exercise())

    assert error.status_code == 403
