"""
notification_provider.py - Notification Provider Abstraction & SMS Gateway Interface
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Provides:
- NotificationProvider: Abstract Base Class for alert delivery channels (SMS, APP, SATELLITE)
- SmsNotificationProvider: Configurable, production-ready SMS gateway integration
- Phone number masking utility ensuring zero PII leak in logs (+9198******10)
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests


def mask_phone_number(phone: Optional[str]) -> str:
    """
    Masks phone numbers for structured logs to prevent PII exposure.
    Example: '+919876543210' -> '+9198******10'
    """
    if not phone:
        return "[UNKNOWN_PHONE]"

    s = str(phone).strip()
    if len(s) < 6:
        return "***"

    prefix_len = 5 if len(s) >= 10 else 2
    suffix_len = 2
    masked_count = max(3, len(s) - prefix_len - suffix_len)
    return f"{s[:prefix_len]}{'*' * masked_count}{s[-suffix_len:]}"


class NotificationProvider(ABC):
    """
    Abstract interface for dispatching maritime alerts across multi-modal channels.
    Designed for cellular SMS with clean extension hooks for future offshore SATELLITE and in-app pushes.
    """

    @abstractmethod
    def send_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """Dispatches an SMS alert to the recipient phone number."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the provider credentials are fully configured."""
        pass


class SmsNotificationProvider(NotificationProvider):
    """
    Configurable SMS notification adapter.
    Reads credentials from environment:
    - SMS_PROVIDER: Provider name (e.g. 'twilio', 'fast2sms', 'msg91', 'custom')
    - SMS_API_KEY: Provider authentication secret
    - SMS_SENDER_ID: Sender ID / caller header
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        sender_id: Optional[str] = None,
    ):
        self._provider = (provider or os.getenv("SMS_PROVIDER", "")).strip().lower()
        self._api_key = (api_key or os.getenv("SMS_API_KEY", "")).strip()
        self._sender_id = (sender_id or os.getenv("SMS_SENDER_ID", "ORCA-ALERT")).strip()

    def is_configured(self) -> bool:
        """Checks whether SMS provider has an active API key configured."""
        api_key = os.getenv("SMS_API_KEY", self._api_key).strip()
        return bool(api_key)

    def get_provider_name(self) -> str:
        return os.getenv("SMS_PROVIDER", self._provider).strip().lower() or "generic_sms"

    def send_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """
        Dispatches SMS via configured provider.
        If unconfigured:
        - Does NOT pretend SMS was sent
        - Does NOT generate fake delivery IDs
        - Marks status as NOT_CONFIGURED
        """
        masked = mask_phone_number(phone_number)

        if not self.is_configured():
            print(f"[Alert System] SMS_FAILED: SMS provider not configured. Delivery skipped for {masked}")
            return {
                "success": False,
                "status": "NOT_CONFIGURED",
                "provider_message_id": None,
                "error": "SMS provider credentials not configured in server environment."
            }

        provider = self.get_provider_name()
        api_key = os.getenv("SMS_API_KEY", self._api_key).strip()
        sender_id = os.getenv("SMS_SENDER_ID", self._sender_id).strip()

        try:
            # Example Generic REST Dispatch implementation
            # Adapts cleanly to provider specifications via POST payload
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "to": phone_number,
                "message": message,
                "sender": sender_id
            }

            # In production, provider-specific URLs are loaded; for demonstration, handle generic hook
            provider_endpoint = os.getenv("SMS_ENDPOINT_URL")
            if not provider_endpoint:
                print(f"[Alert System] SMS_FAILED: No SMS_ENDPOINT_URL set for provider '{provider}'. Target: {masked}")
                return {
                    "success": False,
                    "status": "NOT_CONFIGURED",
                    "provider_message_id": None,
                    "error": f"Provider '{provider}' configured without SMS_ENDPOINT_URL."
                }

            resp = requests.post(provider_endpoint, json=payload, headers=headers, timeout=10)
            if resp.status_code in (200, 201, 202):
                resp_json = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                provider_msg_id = resp_json.get("message_id") or resp_json.get("sid") or f"sms_{resp.status_code}"
                print(f"[Alert System] SMS_SENT to {masked} | Provider ID: {provider_msg_id}")
                return {
                    "success": True,
                    "status": "SENT",
                    "provider_message_id": provider_msg_id,
                    "provider": provider
                }
            else:
                err_text = resp.text[:200]
                print(f"[Alert System] SMS_FAILED for {masked} | HTTP {resp.status_code}: {err_text}")
                return {
                    "success": False,
                    "status": "FAILED",
                    "provider_message_id": None,
                    "error": f"Provider returned HTTP {resp.status_code}: {err_text}"
                }

        except Exception as e:
            print(f"[Alert System] SMS_FAILED for {masked} | Exception: {str(e)}")
            return {
                "success": False,
                "status": "FAILED",
                "provider_message_id": None,
                "error": str(e)
            }


# Global Singleton SMS Provider
sms_provider = SmsNotificationProvider()

