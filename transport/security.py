"""
transport/security.py - Security, Authentication & Payload Validation for Satellite Gateway
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Provides:
- Constant-time API Key verification
- HMAC-SHA256 signature verification for webhook callbacks
- Strict payload size boundaries (preventing memory exhaustion / sat bloat)
- Timestamp replay protection
- Credential redaction for clean security logs
"""

import os
import hmac
import time
import hashlib
from typing import Optional, Union


def get_configured_api_key() -> Optional[str]:
    key = os.getenv("SATELLITE_GATEWAY_API_KEY", "").strip()
    return key if key else None


def get_configured_webhook_secret() -> Optional[str]:
    secret = os.getenv("SATELLITE_WEBHOOK_SECRET", "").strip()
    return secret if secret else None


def get_max_payload_size() -> int:
    try:
        return int(os.getenv("SATELLITE_MAX_PAYLOAD_SIZE", "4096"))
    except ValueError:
        return 4096


def validate_api_key(provided_key: Optional[str]) -> bool:
    """
    Validates the provided API key against SATELLITE_GATEWAY_API_KEY using constant-time comparison.
    If no key is configured on the server, requests are permitted (open for development).
    """
    expected_key = get_configured_api_key()
    if not expected_key:
        return True  # No auth requirement configured on server
    if not provided_key:
        return False
    # Support 'Bearer <key>' or raw '<key>'
    clean_provided = provided_key.replace("Bearer ", "").replace("bearer ", "").strip()
    return hmac.compare_digest(clean_provided, expected_key)


def validate_hmac_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    """
    Validates the HMAC-SHA256 signature of an incoming webhook payload using SATELLITE_WEBHOOK_SECRET.
    Signature header format can be raw hex or 'sha256=<hex>'.
    """
    secret = get_configured_webhook_secret()
    if not secret:
        return True  # Webhook secret not configured
    if not signature_header:
        return False

    clean_sig = signature_header.strip()
    if clean_sig.startswith("sha256="):
        clean_sig = clean_sig.split("=", 1)[1]

    expected_sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(clean_sig.lower(), expected_sig.lower())


def validate_payload_size(raw_body: bytes, max_bytes: Optional[int] = None) -> bool:
    """
    Ensures body does not exceed the allowed maximum payload size.
    Returns True if size is within limits.
    """
    limit = max_bytes if max_bytes is not None else get_max_payload_size()
    return len(raw_body) <= limit


def validate_timestamp(timestamp: Union[int, float], max_skew_seconds: float = 86400.0) -> bool:
    """
    Replay protection: ensures timestamp is not radically skewed into the future or far past.
    Allows up to 24 hours of latency/skew to account for vessel terminal queueing and high satellite latency.
    """
    try:
        ts = float(timestamp)
        now = time.time()
        # Allow +/- max_skew_seconds
        return abs(now - ts) <= max_skew_seconds
    except Exception:
        return False


def redact_secrets(message: str) -> str:
    """
    Strips potential API keys and secrets from log strings.
    """
    redacted = message
    api_key = get_configured_api_key()
    if api_key and len(api_key) > 4:
        redacted = redacted.replace(api_key, "[REDACTED_API_KEY]")
    secret = get_configured_webhook_secret()
    if secret and len(secret) > 4:
        redacted = redacted.replace(secret, "[REDACTED_SECRET]")
    return redacted

