import os
from typing import Optional

from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

_client: Optional[Client] = None


def get_supabase() -> Client:
    """
    Service-role Supabase client (singleton). Bypasses RLS — used only by
    backend-internal writes made on behalf of whichever user is calling
    (their identity is stamped onto rows explicitly, e.g. user_id columns).
    """
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "Supabase service client is not configured "
                "(SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing)"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client
