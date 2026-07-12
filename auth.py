import os
from typing import Optional

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Unused by verify_supabase_token (Supabase signs access tokens with ES256,
# verified below via JWKS, not a shared HS256 secret) — kept in case
# anything else in the codebase still reads it.
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

_bearer_scheme = HTTPBearer(auto_error=False)

# Cached once at module load rather than per-request; PyJWKClient handles its
# own key caching/refresh internally.
_jwks_client: Optional[PyJWKClient] = (
    PyJWKClient(f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json")
    if SUPABASE_URL
    else None
)


class CurrentUser(BaseModel):
    id: str
    email: Optional[str] = None


def verify_supabase_token(authorization_header: Optional[str]) -> CurrentUser:
    """
    Validate a Supabase-issued access token (the frontend session's JWT) and
    return the authenticated user.

    Supabase signs access tokens with ES256 (asymmetric) — verified here
    against the project's public JWKS keys, not a shared secret.

    `authorization_header` is the raw `Authorization` header value, e.g. "Bearer <jwt>".
    Raises HTTPException(401) if the header is missing/malformed or the token
    is invalid, expired, or was not issued by this Supabase project.
    """
    if not SUPABASE_URL or _jwks_client is None:
        raise HTTPException(status_code=500, detail="Supabase auth is not configured on the server")

    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization_header.split(" ", 1)[1].strip()

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
            issuer=f"{SUPABASE_URL.rstrip('/')}/auth/v1",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {e}")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token payload missing subject")

    return CurrentUser(id=user_id, email=payload.get("email"))


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> CurrentUser:
    """FastAPI dependency: resolves the authenticated user from the request's Bearer token."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return verify_supabase_token(f"Bearer {credentials.credentials}")


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[CurrentUser]:
    """
    FastAPI dependency for PUBLIC endpoints that want to know the caller's
    identity *if* they happen to be logged in, without requiring auth.

    Returns the authenticated user when a valid Bearer token is present,
    otherwise returns None (missing header, malformed header, or an
    invalid/expired token) — never raises, so the endpoint stays public.
    """
    if credentials is None:
        return None
    try:
        return verify_supabase_token(f"Bearer {credentials.credentials}")
    except HTTPException:
        return None
