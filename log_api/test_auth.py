import asyncio
from datetime import datetime, timedelta, timezone

import jwt
from fastapi.security import HTTPAuthorizationCredentials

from log_api.auth import AuthConfig, SupabaseJWTVerifier


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
