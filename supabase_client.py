"""
supabase_client.py - Project ORCA Supabase Client Initialization & Health Management
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Initializes the Supabase client using the server-side service-role secret key.
Strict Zero-Credential Security:
- SUPABASE_URL defaults to https://fdvkmfxietjubbzujntz.supabase.co
- SUPABASE_SERVICE_ROLE_KEY is read strictly from environment variables (.env)
- No secret key is hardcoded or exposed to client apps
"""

import os
import threading
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

DEFAULT_SUPABASE_URL = "https://fdvkmfxietjubbzujntz.supabase.co"

_client_lock = threading.Lock()
_supabase_client = None


def get_supabase_url() -> str:
    """Returns configured Supabase URL, defaulting to official project endpoint."""
    url = os.getenv("SUPABASE_URL", "").strip()
    return url if url else DEFAULT_SUPABASE_URL


def get_supabase_service_role_key() -> Optional[str]:
    """
    Reads the Supabase service-role key from environment.
    Supports both SUPABASE_SERVICE_ROLE_KEY and SUPABASE_KEY.
    MUST NOT have a hardcoded default.
    """
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_KEY", "").strip()
    return key if key else None


def is_supabase_configured() -> bool:
    """Returns True if the Supabase service role key is present in environment."""
    return get_supabase_service_role_key() is not None


def get_supabase_client():
    """
    Returns the singleton Supabase client instance.
    Raises RuntimeError with a clear configuration message if the key is missing.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    with _client_lock:
        if _supabase_client is not None:
            return _supabase_client

        url = get_supabase_url()
        key = get_supabase_service_role_key()

        if not key:
            raise RuntimeError(
                "Supabase service role key is not configured on the ORCA server. "
                "Please set the SUPABASE_SERVICE_ROLE_KEY environment variable in your .env file."
            )

        try:
            from supabase import create_client, Client
            _supabase_client = create_client(url, key)
            print(f"[Supabase Client] Successfully initialized connection to: {url}")
            return _supabase_client
        except Exception as e:
            print(f"[Supabase Client Error] Failed to initialize Supabase client: {e}")
            raise


def reset_supabase_client():
    """Resets the singleton instance (useful during testing or configuration reloads)."""
    global _supabase_client
    with _client_lock:
        _supabase_client = None


def check_supabase_health() -> Dict[str, Any]:
    """
    Performs a lightweight health check against the Supabase database.
    Does not crash if Supabase is offline or unconfigured.
    """
    url = get_supabase_url()
    if not is_supabase_configured():
        return {
            "configured": False,
            "status": "NOT_CONFIGURED",
            "url": url,
            "project_url": url,
            "notice": "SUPABASE_SERVICE_ROLE_KEY is not set in environment."
        }

    try:
        client = get_supabase_client()
        # Probe 'users' table without reading full rows
        client.table("users").select("id").limit(1).execute()
        return {
            "configured": True,
            "status": "CONNECTED",
            "url": url,
            "project_url": url,
            "tables": ["users", "user_locations", "alerts", "alert_deliveries"]
        }
    except Exception as e:
        return {
            "configured": True,
            "status": "ERROR",
            "url": url,
            "project_url": url,
            "error": str(e)
        }

