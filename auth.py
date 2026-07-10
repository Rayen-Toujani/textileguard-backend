import os
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

_bearer_scheme = HTTPBearer(auto_error=False)


class CurrentUser(BaseModel):
    id: str
    email: Optional[str] = None


def verify_supabase_token(authorization_header: Optional[str]) -> CurrentUser:
    """
    Validate a Supabase-issued access token (the frontend session's JWT) and
    return the authenticated user.

    `authorization_header` is the raw `Authorization` header value, e.g. "Bearer <jwt>".
    Raises HTTPException(401) if the header is missing/malformed or the token
    is invalid, expired, or was not issued by this Supabase project.
    """
    if not SUPABASE_URL or not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="Supabase auth is not configured on the server")

    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization_header.split(" ", 1)[1].strip()

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
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
