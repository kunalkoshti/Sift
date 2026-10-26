"""Supabase access-token verification for the log API.

The browser authenticates with Supabase Auth.  The API only needs to verify the
resulting access token; it does not need Google-specific authentication code.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from jwt.exceptions import PyJWKClientError
from jwt import PyJWKClient


bearer_scheme = HTTPBearer(auto_error=False)


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError("AUTH_REQUIRED must be true or false")


@dataclass(frozen=True)
class AuthConfig:
    required: bool
    jwks_url: str | None
    issuer: str | None
    audience: str
    jwt_secret: str | None

    @classmethod
    def from_env(cls) -> "AuthConfig":
        required = _as_bool(os.getenv("AUTH_REQUIRED"), default=False)
        supabase_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        jwks_url = os.getenv("SUPABASE_JWKS_URL") or (
            f"{supabase_url}/auth/v1/.well-known/jwks.json" if supabase_url else None
        )
        issuer = os.getenv("SUPABASE_JWT_ISSUER") or (
            f"{supabase_url}/auth/v1" if supabase_url else None
        )
        audience = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")
        jwt_secret = os.getenv("SUPABASE_JWT_SECRET") or None

        if required and not ((jwks_url and issuer) or jwt_secret):
            raise RuntimeError(
                "AUTH_REQUIRED=true requires SUPABASE_URL (or SUPABASE_JWKS_URL "
                "and SUPABASE_JWT_ISSUER), or SUPABASE_JWT_SECRET"
            )
        return cls(required, jwks_url, issuer, audience, jwt_secret)


class SupabaseJWTVerifier:
    """FastAPI dependency that validates Supabase bearer tokens.

    New Supabase projects normally use asymmetric signing keys, which are
    verified through the project's JWKS endpoint.  ``SUPABASE_JWT_SECRET`` is
    retained as an explicit fallback for projects still using legacy HS256
    signing keys.
    """

    def __init__(self, config: AuthConfig):
        self.config = config
        self._jwks_client = (
            PyJWKClient(config.jwks_url)
            if config.jwks_url and not config.jwt_secret
            else None
        )

    async def __call__(
        self,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> dict[str, Any] | None:
        if not self.config.required:
            return None
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid Supabase access token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            # JWT signature validation is local. Keeping this synchronous also
            # avoids leaving executor threads behind during short-lived test
            # clients and Cloud Run startup probes.
            claims = self._decode(credentials.credentials)
        except (InvalidTokenError, PyJWKClientError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="The Supabase access token is invalid or expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        return claims

    def _decode(self, token: str) -> dict[str, Any]:
        if self.config.jwt_secret:
            return jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=["HS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
            )

        if self._jwks_client is None or not self.config.issuer:
            raise RuntimeError("Supabase JWT verification is not configured")
        signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256", "EdDSA"],
            audience=self.config.audience,
            issuer=self.config.issuer,
        )


def _csv_values(name: str) -> frozenset[str]:
    return frozenset(
        value.strip().casefold()
        for value in (os.getenv(name) or "").split(",")
        if value.strip()
    )


def create_admin_dependency(auth_dependency: Any):
    """Authorize a private admin endpoint using verified Supabase claims."""

    allowed_ids = _csv_values("ADMIN_USER_IDS")
    allowed_emails = _csv_values("ADMIN_EMAILS")

    async def dependency(
        claims: dict[str, Any] | None = Depends(auth_dependency),
    ) -> dict[str, Any]:
        if not claims:
            raise HTTPException(
                status_code=403,
                detail="Admin access requires an authenticated account",
            )
        user_id = str(claims.get("sub") or claims.get("user_id") or "").casefold()
        email = str(claims.get("email") or "").casefold()
        if (user_id and user_id in allowed_ids) or (email and email in allowed_emails):
            return claims
        raise HTTPException(
            status_code=403,
            detail="This account is not authorized to view production metrics",
        )

    return dependency
