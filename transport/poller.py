"""
transport/poller.py - Satellite Gateway Polling Worker
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Provides a resilient background polling worker for satellite provider gateways
that do not support push webhooks and instead require the ORCA server to pull
queued satellite messages.
"""

import os
import time
import threading
from typing import Optional, Callable, Dict, Any
import requests

from transport.models import SatelliteMessage
from transport.satellite import gateway_adapter
from transport.adapter import OrcaRequestAdapter, OrcaResponseAdapter
from transport.idempotency import idempotency_store
from transport.security import redact_secrets


class SatelliteGatewayPoller:
    """
    Optional background polling thread for pull-based satellite gateway architectures.
    Only starts when satellite integration is enabled and a polling URL is configured.
    """

    def __init__(
        self,
        poll_url: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        processor_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self._poll_url = poll_url or os.getenv("SATELLITE_POLL_URL", "")
        try:
            self._interval = interval_seconds or int(os.getenv("SATELLITE_POLL_INTERVAL_SECONDS", "60"))
        except ValueError:
            self._interval = 60
        self._processor_callback = processor_callback
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Starts the background poller thread if enabled."""
        if not gateway_adapter.is_enabled():
            print("[Satellite Poller] Poller not started: Satellite integration is DISABLED.")
            return

        target_url = self._poll_url or gateway_adapter.get_gateway_url()
        if not target_url:
            print("[Satellite Poller] Poller not started: No polling URL configured.")
            return

        if self.is_running():
            print("[Satellite Poller] Poller is already active.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_poll_loop, daemon=True, name="orca-sat-poller")
        self._thread.start()
        print(f"[Satellite Poller] Started background polling worker (Interval: {self._interval}s)")

    def stop(self):
        """Stops the poller thread."""
        if self.is_running():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            print("[Satellite Poller] Background polling worker stopped.")

    def _run_poll_loop(self):
        target_url = self._poll_url or f"{gateway_adapter.get_gateway_url().rstrip('/')}/messages"
        headers = {}
        api_key = gateway_adapter.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-Satellite-Api-Key"] = api_key

        while not self._stop_event.is_set():
            try:
                # Poll gateway
                res = requests.get(target_url, headers=headers, timeout=15)
                if res.status_code == 200:
                    messages = res.json()
                    if isinstance(messages, list):
                        for raw_msg in messages:
                            self._process_polled_message(raw_msg)
            except Exception as e:
                print(f"[Satellite Poller Warning] Poll cycle error: {redact_secrets(str(e))}")

            # Sleep in short increments to allow graceful shutdown
            for _ in range(self._interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

    def _process_polled_message(self, raw_msg: Dict[str, Any]):
        try:
            sat_msg = SatelliteMessage(**raw_msg)
            message_id = sat_msg.message_id

            # Check idempotency
            record, is_new = idempotency_store.register(message_id, sat_msg.model_dump())
            if not is_new and record.status.value == "COMPLETED":
                print(f"[Satellite Poller] Skipping duplicate message_id: {message_id}")
                gateway_adapter.acknowledge(message_id)
                return

            idempotency_store.mark_processing(message_id)

            if self._processor_callback:
                params = OrcaRequestAdapter.to_core_params(sat_msg)
                orca_result = self._processor_callback(params)
                compact_resp = OrcaResponseAdapter.to_compact_response(sat_msg, orca_result)
                idempotency_store.mark_completed(message_id, compact_resp)
                gateway_adapter.send_compact_response(compact_resp)
                gateway_adapter.acknowledge(message_id)
        except Exception as e:
            print(f"[Satellite Poller Error] Failed to process polled message: {redact_secrets(str(e))}")

