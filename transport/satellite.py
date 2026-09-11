"""
transport/satellite.py - Satellite Transport Abstraction & Gateway Adapter
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Provides:
1. `SatelliteTransport`: Abstract Base Class defining transport boundaries.
2. `SatelliteGatewayAdapter`: Configurable production-ready adapter for communicating with
   external satellite ground infrastructure / provider APIs (e.g. Iridium, Inmarsat, NavIC/GSAT gateway).
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests

from transport.models import SatelliteCompactResponse
from transport.security import redact_secrets


class SatelliteTransport(ABC):
    """
    Abstract interface for satellite communication transports.
    Decouples ORCA server from specific vendor protocols, radio modems, or ground station networks.
    """

    @abstractmethod
    def send(self, message: Any) -> bool:
        """Transmits a message to the satellite communication channel / gateway."""
        pass

    @abstractmethod
    def receive(self) -> Optional[Any]:
        """Receives a message from the satellite communication channel / gateway."""
        pass

    @abstractmethod
    def acknowledge(self, message_id: str) -> bool:
        """Acknowledges successful message receipt/delivery to the gateway."""
        pass


class SatelliteGatewayAdapter(SatelliteTransport):
    """
    Configurable gateway adapter connecting the ORCA server to an external
    satellite provider's ground station or maritime communication gateway.

    Configuration via Environment:
    - SATELLITE_GATEWAY_ENABLED: bool ("true" | "false", default "false")
    - SATELLITE_GATEWAY_URL: target gateway endpoint for outbound responses
    - SATELLITE_GATEWAY_API_KEY: optional provider authentication token
    - SATELLITE_MESSAGE_TIMEOUT: transmission timeout in seconds (default 30)
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        gateway_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        env_enabled = os.getenv("SATELLITE_GATEWAY_ENABLED", "false").lower() in ("true", "1", "yes")
        self._enabled = enabled if enabled is not None else env_enabled
        self._gateway_url = gateway_url or os.getenv("SATELLITE_GATEWAY_URL", "").strip() or None
        self._api_key = api_key or os.getenv("SATELLITE_GATEWAY_API_KEY", "").strip() or None

        try:
            self._timeout = timeout or int(os.getenv("SATELLITE_MESSAGE_TIMEOUT", "30"))
        except ValueError:
            self._timeout = 30

    def is_enabled(self) -> bool:
        """Returns whether satellite gateway integration is enabled."""
        # Re-check env dynamically if not explicitly overridden
        if "SATELLITE_GATEWAY_ENABLED" in os.environ:
            return os.getenv("SATELLITE_GATEWAY_ENABLED", "false").lower() in ("true", "1", "yes")
        return self._enabled

    def get_gateway_url(self) -> Optional[str]:
        return os.getenv("SATELLITE_GATEWAY_URL", "").strip() or self._gateway_url

    def get_api_key(self) -> Optional[str]:
        return os.getenv("SATELLITE_GATEWAY_API_KEY", "").strip() or self._api_key

    def send(self, message: Any) -> bool:
        """Generic send implementation complying with SatelliteTransport ABC."""
        if isinstance(message, SatelliteCompactResponse):
            res = self.send_compact_response(message)
            return res.get("success", False)
        return False

    def receive(self) -> Optional[Any]:
        """Push-model gateways use webhooks; polling adapters override this."""
        return None

    def acknowledge(self, message_id: str) -> bool:
        """
        Sends an ACK packet to the satellite ground gateway if an ACK endpoint is configured.
        """
        if not self.is_enabled():
            return False

        gateway_url = self.get_gateway_url()
        if not gateway_url:
            return True  # Acknowledged in-memory without remote gateway callback

        ack_url = f"{gateway_url.rstrip('/')}/ack"
        headers = {"Content-Type": "application/json"}
        api_key = self.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-Satellite-Api-Key"] = api_key

        payload = {
            "protocol_version": 1,
            "message_id": message_id,
            "status": "ACK",
            "timestamp": int(time.time()),
        }

        try:
            resp = requests.post(ack_url, json=payload, headers=headers, timeout=self._timeout)
            if resp.status_code in (200, 202, 204):
                print(f"[Satellite Gateway] Acknowledged message_id: {message_id}")
                return True
            print(f"[Satellite Gateway Warning] Ack gateway returned HTTP {resp.status_code} for message_id: {message_id}")
            return False
        except Exception as e:
            print(f"[Satellite Gateway Error] Ack transmission failed for message_id: {message_id}: {redact_secrets(str(e))}")
            return False

    def send_compact_response(
        self,
        response: SatelliteCompactResponse,
        callback_url: Optional[str] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Transmits the bandwidth-minimized compact ORCA advisory back to the satellite gateway.
        Handles retries, network latency, and non-blocking failure reporting.
        """
        message_id = response.message_id

        if not self.is_enabled():
            print(f"[Satellite Gateway] Send bypassed: Satellite integration is DISABLED. message_id: {message_id}")
            return {
                "success": False,
                "reason": "SATELLITE_GATEWAY_DISABLED",
                "message_id": message_id,
            }

        target_url = callback_url or self.get_gateway_url()
        if not target_url:
            # Synchronous direct response path (HTTP request caller gets the response directly)
            print(f"[Satellite Gateway] Direct HTTP response path (no external outbound gateway URL configured). message_id: {message_id}")
            return {
                "success": True,
                "reason": "DIRECT_HTTP_PATH",
                "message_id": message_id,
                "response": response.model_dump(),
            }

        headers = {
            "Content-Type": "application/json",
            "X-ORCA-Message-Id": message_id,
            "X-ORCA-Protocol-Version": str(response.protocol_version),
        }
        api_key = self.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-Satellite-Api-Key"] = api_key

        payload = response.model_dump()

        attempt = 0
        last_error = None
        while attempt <= max_retries:
            attempt += 1
            try:
                print(f"[Satellite Gateway] Outbound send attempt {attempt}/{max_retries + 1} for message_id: {message_id}")
                res = requests.post(target_url, json=payload, headers=headers, timeout=self._timeout)
                if res.status_code in (200, 201, 202, 204):
                    print(f"[Satellite Gateway] Gateway send SUCCESS: message_id: {message_id} | HTTP {res.status_code}")
                    return {
                        "success": True,
                        "status_code": res.status_code,
                        "message_id": message_id,
                        "attempts": attempt,
                    }
                else:
                    last_error = f"Gateway returned HTTP {res.status_code}: {res.text[:200]}"
                    print(f"[Satellite Gateway Warning] Outbound send attempt {attempt} failed: {last_error}")
            except requests.exceptions.RequestException as re_err:
                last_error = str(re_err)
                print(f"[Satellite Gateway Warning] Outbound network error attempt {attempt}: {redact_secrets(last_error)}")

            if attempt <= max_retries:
                time.sleep(1.0 * attempt)

        print(f"[Satellite Gateway Error] Outbound send FAILURE after {attempt} attempts for message_id: {message_id}: {redact_secrets(str(last_error))}")
        return {
            "success": False,
            "reason": "GATEWAY_TRANSMISSION_FAILED",
            "error": redact_secrets(str(last_error)),
            "message_id": message_id,
            "attempts": attempt,
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic status for health endpoints."""
        enabled = self.is_enabled()
        url = self.get_gateway_url()
        has_url = bool(url)
        has_key = bool(self.get_api_key())

        if not enabled:
            status_str = "DISABLED"
        elif has_url:
            status_str = "READY"
        else:
            status_str = "READY_DIRECT_ONLY"

        return {
            "enabled": enabled,
            "status": status_str,
            "adapter": "SatelliteGatewayAdapter",
            "gateway_url_configured": has_url,
            "auth_configured": has_key,
            "timeout_seconds": self._timeout,
            "notice": (
                "Offshore satellite connectivity requires vessel terminal hardware and ground gateway provider. "
                "Direct API mode is active."
            ),
        }


# Global Singleton Gateway Adapter
gateway_adapter = SatelliteGatewayAdapter()

