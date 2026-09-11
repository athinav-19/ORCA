"""
sms_provider.py - SMS Provider Abstraction & Multi-Gateway Adapter
Project ORCA: Marine Multi-Agent Intelligence Backend
ISRO SIH Problem Statement 176: Coastal Early Warning System

Provides a clean, modular, and isolated interface for dispatching cellular SMS
emergency alerts to offshore coastal stakeholders, fishermen, and maritime operators.

Architecture:
Alert System (alert_service.py)
      ↓
SMSProvider abstraction (BaseSmsProvider)
      ↓
Configured Gateway Implementation (Twilio / Fast2SMS / MSG91 / Generic REST / Unconfigured)

Three SMS Modes:
- SMS_MODE=disabled: Default development safe mode; zero external calls, no real SMS.
- SMS_MODE=test: Development test mode; only delivers to whitelisted numbers in SMS_TEST_PHONE_NUMBERS.
- SMS_MODE=live: Production mode; dispatches live alerts via configured gateway.

Security Rules:
- Zero credential logging or exposure.
- Phone numbers are masked in all logs (e.g. +9198******10).
- If unconfigured, does not fake delivery or generate bogus message IDs.
- Honest state transitions: PENDING -> SENT / DELIVERED / FAILED.
"""

import os
import re
import time
import base64
import json
import urllib.request
import urllib.error
import urllib.parse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List


def mask_phone_number(phone: Optional[str]) -> str:
    """
    Masks phone numbers for logs and audit trails to prevent PII exposure.
    Example: '+919876543210' -> '+9198******10'
    """
    if not phone:
        return "[NO_PHONE]"
    s = str(phone).strip()
    if len(s) < 7:
        return "***"
    prefix_len = 5 if len(s) >= 10 else 2
    suffix_len = 2
    masked_count = max(3, len(s) - prefix_len - suffix_len)
    return f"{s[:prefix_len]}{'*' * masked_count}{s[-suffix_len:]}"


def normalize_and_validate_phone(phone: Optional[str]) -> Tuple[bool, str, Optional[str]]:
    """
    Validates and normalizes phone numbers to standard E.164 format.
    Specialized for Indian mobile numbers (+91), while supporting international E.164.

    Returns:
        (is_valid, normalized_phone, error_message)
    """
    if not phone:
        return False, "", "Phone number is missing or empty."

    raw = str(phone).strip()
    # Check for invalid non-phone characters (letters, special symbols other than +, -, space, brackets)
    if re.search(r"[a-zA-Z]", raw):
        return False, "", "Invalid phone number format: contains non-numeric characters."

    digits = re.sub(r"[^\d]", "", raw)
    if len(digits) < 10:
        return False, "", f"Invalid phone number format: contains only {len(digits)} digits (minimum 10 required)."

    # If starts with + and has 10-15 digits
    if raw.startswith("+"):
        if 10 <= len(digits) <= 15:
            return True, f"+{digits}", None
        return False, "", f"Invalid E.164 phone number: length must be 10-15 digits."

    # Standard Indian mobile handling
    if len(digits) == 10:
        # Standard 10-digit Indian mobile
        return True, f"+91{digits}", None
    elif len(digits) == 11 and digits.startswith("0"):
        # 11-digit with leading 0 (e.g. 09876543210)
        return True, f"+91{digits[1:]}", None
    elif len(digits) == 12 and digits.startswith("91"):
        # 12-digit Indian with country code without '+'
        return True, f"+{digits}", None
    elif 10 <= len(digits) <= 15:
        # Generic international number missing '+'
        return True, f"+{digits}", None

    return False, "", f"Invalid phone number length: {len(digits)} digits."


def normalize_phone_e164(phone: Optional[str]) -> str:
    """Convenience helper returning normalized string or empty string on error."""
    valid, normalized, _ = normalize_and_validate_phone(phone)
    return normalized if valid else ""


def is_transient_error(error_str: Optional[str], status_code: Optional[int] = None) -> bool:
    """
    Determines whether an SMS dispatch failure is transient (retryable)
    or permanent (non-retryable).
    """
    if status_code is not None:
        if status_code in (408, 429, 500, 502, 503, 504):
            return True
        if 400 <= status_code < 500:
            return False  # Authentication, invalid request, not found -> permanent

    err = (error_str or "").lower()
    transient_keywords = [
        "timeout", "timed out", "connection reset", "connection refused",
        "temporarily unavailable", "try again", "rate limit", "too many requests",
        "service unavailable", "gateway timeout"
    ]
    return any(k in err for k in transient_keywords)


# =====================================================================
# SMS Provider Interface & Gateway Implementations
# =====================================================================

class BaseSmsProvider(ABC):
    """Abstract SMS Gateway Interface."""

    provider_name: str = "base"

    @abstractmethod
    def send(self, phone_number: str, message: str) -> Dict[str, Any]:
        """
        Sends an SMS alert.
        Returns:
            dict with:
                - success: bool
                - status: 'SENT' | 'DELIVERED' | 'FAILED' | 'NOT_CONFIGURED'
                - provider_message_id: str | None
                - error: str | None
                - is_transient: bool
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the required credentials exist in environment."""
        pass


# Alias for clean architecture requirement
SMSProvider = BaseSmsProvider


class TwilioSmsProvider(BaseSmsProvider):
    """Twilio REST API Provider."""

    provider_name: str = "twilio"

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid.strip()
        self.auth_token = auth_token.strip()
        self.from_number = from_number.strip()

    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.from_number)

    def send(self, phone_number: str, message: str) -> Dict[str, Any]:
        masked = mask_phone_number(phone_number)
        if not self.is_configured():
            return {
                "success": False,
                "status": "NOT_CONFIGURED",
                "provider_message_id": None,
                "error": "Twilio Account SID, Auth Token, or From number not configured.",
                "is_transient": False,
            }

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = urllib.parse.urlencode({
            "To": phone_number,
            "From": self.from_number,
            "Body": message
        }).encode("utf-8")

        credentials = f"{self.account_sid}:{self.auth_token}"
        auth_header = f"Basic {base64.b64encode(credentials.encode()).decode()}"

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                sid = body.get("sid")
                t_status = (body.get("status") or "").lower()
                status = "DELIVERED" if t_status == "delivered" else "SENT"
                print(f"[SMS Provider: Twilio] SMS submitted to {masked} | SID: {sid} | Status: {status}")
                return {
                    "success": True,
                    "status": status,
                    "provider_message_id": sid,
                    "error": None,
                    "is_transient": False,
                }
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")[:200]
            print(f"[SMS Provider: Twilio Error] Failed for {masked} (HTTP {he.code}): {err_body}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": f"Twilio HTTP {he.code}: {err_body}",
                "is_transient": is_transient_error(err_body, he.code),
            }
        except Exception as ex:
            print(f"[SMS Provider: Twilio Error] Failed for {masked}: {ex}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": str(ex),
                "is_transient": is_transient_error(str(ex)),
            }


class Fast2SmsProvider(BaseSmsProvider):
    """Fast2SMS Indian Gateway Provider."""

    provider_name: str = "fast2sms"

    def __init__(self, api_key: str, sender_id: str = "ORCA"):
        self.api_key = api_key.strip()
        self.sender_id = sender_id.strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def send(self, phone_number: str, message: str) -> Dict[str, Any]:
        masked = mask_phone_number(phone_number)
        if not self.is_configured():
            return {
                "success": False,
                "status": "NOT_CONFIGURED",
                "provider_message_id": None,
                "error": "Fast2SMS API key not configured.",
                "is_transient": False,
            }

        # Fast2SMS expects 10-digit number for Indian numbers
        digits = re.sub(r"[^\d]", "", phone_number)
        if len(digits) > 10 and digits.startswith("91"):
            digits = digits[2:]

        url = "https://www.fast2sms.com/dev/bulkV2"
        payload = json.dumps({
            "route": "v3",
            "sender_id": self.sender_id,
            "message": message,
            "language": "english",
            "flash": 0,
            "numbers": digits
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "authorization": self.api_key,
                "Content-Type": "application/json"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if body.get("return") is True:
                    req_id = body.get("request_id")
                    print(f"[SMS Provider: Fast2SMS] SMS submitted to {masked} | Request ID: {req_id}")
                    return {
                        "success": True,
                        "status": "SENT",
                        "provider_message_id": req_id,
                        "error": None,
                        "is_transient": False,
                    }
                else:
                    err_msg = str(body.get("message", "Fast2SMS rejected dispatch"))
                    print(f"[SMS Provider: Fast2SMS Error] Failed for {masked}: {err_msg}")
                    return {
                        "success": False,
                        "status": "FAILED",
                        "provider_message_id": None,
                        "error": err_msg,
                        "is_transient": is_transient_error(err_msg),
                    }
        except urllib.error.HTTPError as he:
            err_text = he.read().decode("utf-8", errors="replace")[:200]
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": f"Fast2SMS HTTP {he.code}: {err_text}",
                "is_transient": is_transient_error(err_text, he.code),
            }
        except Exception as ex:
            print(f"[SMS Provider: Fast2SMS Error] Exception for {masked}: {ex}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": str(ex),
                "is_transient": is_transient_error(str(ex)),
            }


class Msg91SmsProvider(BaseSmsProvider):
    """MSG91 Indian Transactional & Alert Gateway Provider."""

    provider_name: str = "msg91"

    def __init__(self, auth_key: str, sender_id: str = "ORCAAL", flow_id: Optional[str] = None):
        self.auth_key = auth_key.strip()
        self.sender_id = sender_id.strip()
        self.flow_id = (flow_id or "").strip()

    def is_configured(self) -> bool:
        return bool(self.auth_key)

    def send(self, phone_number: str, message: str) -> Dict[str, Any]:
        masked = mask_phone_number(phone_number)
        if not self.is_configured():
            return {
                "success": False,
                "status": "NOT_CONFIGURED",
                "provider_message_id": None,
                "error": "MSG91 AuthKey not configured.",
                "is_transient": False,
            }

        # MSG91 flow API or direct SMS API
        digits = re.sub(r"[^\d]", "", phone_number)
        url = "https://control.msg91.com/api/v5/flow/" if self.flow_id else "https://api.msg91.com/api/v2/sendsms"

        headers = {
            "authkey": self.auth_key,
            "content-type": "application/json"
        }

        if self.flow_id:
            payload = json.dumps({
                "template_id": self.flow_id,
                "short_url": "0",
                "recipients": [{"mobiles": digits, "message": message}]
            }).encode("utf-8")
        else:
            payload = json.dumps({
                "sender": self.sender_id,
                "route": "4",
                "country": "91",
                "sms": [{"message": message, "to": [digits]}]
            }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw)
                    msg_id = data.get("message_id") or data.get("request_id") or "msg91_ok"
                    t_type = (data.get("type") or "").lower()
                    if t_type == "error":
                        return {
                            "success": False,
                            "status": "FAILED",
                            "provider_message_id": None,
                            "error": data.get("message", "MSG91 rejected dispatch"),
                            "is_transient": False,
                        }
                except Exception:
                    msg_id = "msg91_ok"
                print(f"[SMS Provider: MSG91] SMS submitted to {masked} | Message ID: {msg_id}")
                return {
                    "success": True,
                    "status": "SENT",
                    "provider_message_id": str(msg_id),
                    "error": None,
                    "is_transient": False,
                }
        except urllib.error.HTTPError as he:
            err_text = he.read().decode("utf-8", errors="replace")[:200]
            print(f"[SMS Provider: MSG91 Error] Failed for {masked} (HTTP {he.code}): {err_text}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": f"MSG91 HTTP {he.code}: {err_text}",
                "is_transient": is_transient_error(err_text, he.code),
            }
        except Exception as ex:
            print(f"[SMS Provider: MSG91 Error] Failed for {masked}: {ex}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": str(ex),
                "is_transient": is_transient_error(str(ex)),
            }


class GenericRestSmsProvider(BaseSmsProvider):
    """Custom REST Webhook / Generic SMS Gateway Provider."""

    provider_name: str = "generic_rest"

    def __init__(self, endpoint_url: str, api_key: Optional[str] = None, sender_id: str = "ORCA"):
        self.endpoint_url = endpoint_url.strip()
        self.api_key = (api_key or "").strip()
        self.sender_id = sender_id.strip()

    def is_configured(self) -> bool:
        return bool(self.endpoint_url)

    def send(self, phone_number: str, message: str) -> Dict[str, Any]:
        masked = mask_phone_number(phone_number)
        if not self.is_configured():
            return {
                "success": False,
                "status": "NOT_CONFIGURED",
                "provider_message_id": None,
                "error": "SMS_ENDPOINT_URL not configured.",
                "is_transient": False,
            }

        payload = json.dumps({
            "to": phone_number,
            "message": message,
            "sender_id": self.sender_id
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.endpoint_url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw)
                    msg_id = data.get("message_id") or data.get("id") or "rest_ok"
                except Exception:
                    msg_id = "rest_ok"
                print(f"[SMS Provider: Generic REST] SMS submitted to {masked} | Message ID: {msg_id}")
                return {
                    "success": True,
                    "status": "SENT",
                    "provider_message_id": str(msg_id),
                    "error": None,
                    "is_transient": False,
                }
        except urllib.error.HTTPError as he:
            err_text = he.read().decode("utf-8", errors="replace")[:200]
            print(f"[SMS Provider: Generic REST Error] Failed for {masked} (HTTP {he.code}): {err_text}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": f"HTTP {he.code}: {err_text}",
                "is_transient": is_transient_error(err_text, he.code),
            }
        except Exception as ex:
            print(f"[SMS Provider: Generic REST Error] Failed for {masked}: {ex}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": str(ex),
                "is_transient": is_transient_error(str(ex)),
            }


class UnconfiguredSmsProvider(BaseSmsProvider):
    """Safe fallback provider when no SMS credentials exist in environment."""

    provider_name: str = "none"

    def is_configured(self) -> bool:
        return False

    def send(self, phone_number: str, message: str) -> Dict[str, Any]:
        masked = mask_phone_number(phone_number)
        print(f"[SMS Provider Notice] SMS gateway unconfigured in .env. Delivery skipped for {masked}.")
        return {
            "success": False,
            "status": "FAILED",
            "provider_message_id": None,
            "error": "SMS provider credentials not configured in server environment.",
            "is_transient": False,
        }


def get_sms_provider() -> BaseSmsProvider:
    """
    Factory function instantiating the active SMS provider based on environment variables.
    """
    provider_name = (os.getenv("SMS_PROVIDER") or "").strip().lower()
    api_key = (os.getenv("SMS_API_KEY") or os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    api_secret = (os.getenv("SMS_API_SECRET") or os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    from_number = (os.getenv("TWILIO_FROM_NUMBER") or os.getenv("SMS_SENDER_ID") or "ORCA-ALERT").strip()
    sender_id = (os.getenv("SMS_SENDER_ID") or "ORCA-ALERT").strip()
    endpoint_url = (os.getenv("SMS_ENDPOINT_URL") or "").strip()
    flow_id = (os.getenv("MSG91_FLOW_ID") or "").strip()

    if provider_name == "twilio" or (api_key.startswith("AC") and api_secret):
        return TwilioSmsProvider(account_sid=api_key, auth_token=api_secret, from_number=from_number)

    if provider_name == "fast2sms" and api_key:
        return Fast2SmsProvider(api_key=api_key, sender_id=sender_id)

    if provider_name == "msg91" and api_key:
        return Msg91SmsProvider(auth_key=api_key, sender_id=sender_id, flow_id=flow_id)

    if endpoint_url:
        return GenericRestSmsProvider(endpoint_url=endpoint_url, api_key=api_key, sender_id=sender_id)

    if api_key and endpoint_url:
        return GenericRestSmsProvider(endpoint_url=endpoint_url, api_key=api_key, sender_id=sender_id)

    return UnconfiguredSmsProvider()


def get_sms_mode() -> str:
    """
    Resolves the active SMS mode: 'disabled', 'test', or 'live'.
    Checks SMS_MODE first, falls back to ALERT_SMS_MODE, defaults to 'disabled'.
    """
    mode = os.getenv("SMS_MODE") or os.getenv("ALERT_SMS_MODE") or "disabled"
    clean = mode.strip().lower()
    return clean if clean in ("disabled", "test", "live") else "disabled"


def get_sms_status() -> Dict[str, Any]:
    """
    Returns public, sanitized status metadata for health checks and startup validation.
    Zero credentials exposed.
    """
    mode = get_sms_mode()
    provider = get_sms_provider()
    master_on = os.getenv("SMS_ALERTS_ENABLED", "true").strip().lower() in ("true", "1", "yes", "on")
    whitelist_raw = os.getenv("SMS_TEST_PHONE_NUMBERS", "+919876543210")
    whitelist = [normalize_phone_e164(p.strip()) for p in whitelist_raw.split(",") if p.strip()]

    return {
        "mode": mode.upper(),
        "provider": provider.provider_name.upper(),
        "is_configured": provider.is_configured(),
        "status": "CONFIGURED" if provider.is_configured() else "NOT CONFIGURED",
        "master_switch": "ON" if master_on else "OFF",
        "test_whitelist_count": len(whitelist),
    }


def print_sms_startup_banner():
    """Prints a clear, secure startup configuration banner to the server terminal."""
    status = get_sms_status()
    print("=" * 60)
    print("[SMS Provider Configuration]")
    print(f"- Mode:     {status['mode']}")
    print(f"- Provider: {status['provider']}")
    print(f"- Status:   {status['status']}")
    print(f"- Master:   {status['master_switch']}")
    if status['mode'] == "LIVE" and not status['is_configured']:
        print("[SMS Provider Warning] SMS_MODE=LIVE is active, but required SMS credentials are not configured in .env.")
        print("[SMS Provider Warning] Real SMS broadcasts will fail safely until credentials are provided.")
    elif status['mode'] == "TEST":
        print("[SMS Provider Notice] TEST mode active. Real SMS restricted to configured whitelist.")
    elif status['mode'] == "DISABLED":
        print("[SMS Provider Notice] DISABLED mode active. Safe development default; zero SMS dispatched.")
    print("=" * 60)


# =====================================================================
# Main Dispatch Interface with Three-Tier Safety Guard & Retry Engine
# =====================================================================

def send_sms(
    phone_number: str,
    message: str,
    max_retries: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main entrypoint for SMS transmission with:
    1. Phone number format validation & E.164 normalization.
    2. Master Kill Switch Guard (SMS_ALERTS_ENABLED).
    3. Three-Tier Safety Mode Guard (disabled | test | live).
    4. Transient Error Retry Policy with exponential backoff.
    5. Masked logging ensuring zero PII leak.
    """
    # 1. Phone number validation
    is_valid, normalized, val_error = normalize_and_validate_phone(phone_number)
    if not is_valid:
        masked = mask_phone_number(phone_number)
        print(f"[SMS Provider Error] Invalid phone number for {masked}: {val_error}")
        return {
            "success": False,
            "status": "FAILED",
            "provider_message_id": None,
            "error": val_error or "Invalid phone number format.",
            "is_transient": False,
        }

    masked = mask_phone_number(normalized)

    # 2. Master Kill Switch Guard
    if os.getenv("SMS_ALERTS_ENABLED", "true").strip().lower() in ("false", "0", "no", "off"):
        print(f"[SMS Safety Guard] SMS skipped for {masked}: SMS_ALERTS_ENABLED is false.")
        return {
            "success": False,
            "status": "FAILED",
            "provider_message_id": None,
            "error": "SMS transmission disabled: SMS_ALERTS_ENABLED=false.",
            "is_transient": False,
        }

    # 3. Three-Tier Safety Mode Guard (disabled | test | live)
    sms_mode = get_sms_mode()

    if sms_mode == "disabled":
        print(f"[SMS Safety Guard] SMS skipped for {masked}: SMS_MODE=disabled (safe development default).")
        return {
            "success": False,
            "status": "FAILED",
            "provider_message_id": None,
            "error": "SMS transmission disabled: SMS_MODE=disabled.",
            "is_transient": False,
        }

    elif sms_mode == "test":
        test_numbers_raw = os.getenv("SMS_TEST_PHONE_NUMBERS", "+919876543210")
        allowed_test_numbers = [normalize_phone_e164(p.strip()) for p in test_numbers_raw.split(",") if p.strip()]
        print(f"[SMS Provider: TEST Mode] Evaluating dispatch for recipient: {masked}")

        if normalized not in allowed_test_numbers:
            print(f"[SMS Safety Guard] SMS skipped for {masked}: Recipient not in configured SMS_TEST_PHONE_NUMBERS (SMS_MODE=test).")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": "SMS skipped: Recipient not in configured test numbers (SMS_MODE=test).",
                "is_transient": False,
            }

    elif sms_mode == "live":
        # In LIVE mode, confirm provider is actually configured
        provider = get_sms_provider()
        if not provider.is_configured():
            print(f"[SMS Provider Warning] LIVE mode requested, but gateway credentials missing. Skipping {masked}.")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": "SMS provider credentials not configured in server environment.",
                "is_transient": False,
            }

    # 4. Safe Retry Execution for Transient Failures
    provider = get_sms_provider()
    retries = max_retries if max_retries is not None else int(os.getenv("SMS_MAX_RETRIES", "2"))
    last_result: Dict[str, Any] = {}

    for attempt in range(retries + 1):
        try:
            last_result = provider.send(normalized, message)
            if last_result.get("success") or not last_result.get("is_transient", False):
                # Succeeded or permanent error -> do not retry
                return last_result

            # If transient error and retries remain
            if attempt < retries:
                backoff_sec = 0.5 * (2 ** attempt)
                print(f"[SMS Provider Retry] Transient error for {masked} (attempt {attempt+1}/{retries+1}). Retrying in {backoff_sec:.1f}s...")
                time.sleep(backoff_sec)
        except Exception as exc:
            err_str = str(exc)
            transient = is_transient_error(err_str)
            last_result = {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": err_str,
                "is_transient": transient,
            }
            if not transient or attempt >= retries:
                return last_result
            backoff_sec = 0.5 * (2 ** attempt)
            print(f"[SMS Provider Retry] Exception for {masked} (attempt {attempt+1}/{retries+1}). Retrying in {backoff_sec:.1f}s...")
            time.sleep(backoff_sec)

    return last_result
