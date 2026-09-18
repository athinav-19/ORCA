"""
test_satellite_gateway.py - Offshore Satellite Communication Gateway Test Suite
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Validates all 10 offshore satellite communication test cases:
- TEST 1: Valid satellite query execution (HTTP 200, message_id echo, compact response)
- TEST 2: Invalid latitude boundary rejection (-90 <= lat <= 90) -> HTTP 422
- TEST 3: Invalid longitude boundary rejection (-180 <= lon <= 180) -> HTTP 422
- TEST 4: Missing or invalid message_id rejection -> HTTP 422
- TEST 5: Duplicate message_id idempotency handling (cached=True, zero redundant agent runs)
- TEST 6: Satellite disabled mode handling (SATELLITE_GATEWAY_ENABLED=false) -> HTTP 503
- TEST 7: Gateway unavailable outbound resilience & logging without server crash
- TEST 8: Normal /api/chat backward-compatibility & zero-regression verification
- TEST 9: Satellite simulator full round-trip verification
- TEST 10: Oversized satellite payload guard (SATELLITE_MAX_PAYLOAD_SIZE) -> HTTP 413
"""

import os
import sys
import json
import time
import uuid
import asyncio
import unittest
import httpx

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import application and transport modules
from server import app
from transport import (
    gateway_adapter,
    idempotency_store,
    SatelliteMessage,
    SatellitePayload,
    SatelliteCompactResponse,
)
from satellite_simulator import build_simulated_satellite_packet


class SyncASGIClient:
    """Synchronous test client adapter for httpx.AsyncClient with ASGITransport."""

    def __init__(self, asgi_app):
        self.app = asgi_app
        self.base_url = "http://testserver"

    def get(self, url: str, **kwargs) -> httpx.Response:
        async def _call():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=self.base_url) as c:
                return await c.get(url, **kwargs)
        return asyncio.run(_call())

    def post(self, url: str, **kwargs) -> httpx.Response:
        async def _call():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=self.base_url) as c:
                return await c.post(url, **kwargs)
        return asyncio.run(_call())


class TestOffshoreSatelliteGateway(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = SyncASGIClient(app)

    def setUp(self):
        # Default test configuration: satellite gateway enabled in FAST_DEMO_MODE
        os.environ["SATELLITE_GATEWAY_ENABLED"] = "true"
        os.environ["FAST_DEMO_MODE"] = "true"
        idempotency_store.clear()

    def tearDown(self):
        idempotency_store.clear()

    def test_01_valid_satellite_query(self):
        """TEST 1: Valid satellite message returns HTTP 200, matching message_id, and compact response."""
        msg_id = f"test-sat-{int(time.time())}"
        payload = {
            "protocol_version": 1,
            "message_id": msg_id,
            "message_type": "ORCA_QUERY",
            "timestamp": int(time.time()),
            "payload": {
                "query": "Is it safe to fish here tomorrow morning?",
                "latitude": 8.7642,
                "longitude": 78.1348,
                "persona": "FISHERMAN",
                "language": "en",
                "speed_knots": 4.2,
                "heading_degrees": 120.0
            }
        }

        resp = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp.status_code, 200, f"Expected HTTP 200, got: {resp.text}")

        data = resp.json()
        self.assertEqual(data.get("protocol_version"), 1)
        self.assertEqual(data.get("message_id"), msg_id)
        self.assertIn("status", data)
        self.assertIn("risk", data)
        self.assertIn("advisory", data)
        self.assertGreater(len(data["advisory"]), 10)
        self.assertEqual(round(data.get("latitude"), 4), 8.7642)
        self.assertEqual(round(data.get("longitude"), 4), 78.1348)
        self.assertIn("timestamp", data)
        # Verify compact nature: no heavy geojson, base64 audio, or debug dump
        self.assertNotIn("safe_sea_route", data)
        self.assertNotIn("features", data)
        self.assertNotIn("audio_payload_base64", data)
        print("\n✓ TEST 1 PASSED: Valid satellite query processed successfully.")

    def test_02_invalid_latitude(self):
        """TEST 2: Out of range latitude (-90 to 90) returns HTTP 422 validation error."""
        payload = {
            "protocol_version": 1,
            "message_id": "test-lat-invalid",
            "message_type": "ORCA_QUERY",
            "timestamp": int(time.time()),
            "payload": {
                "query": "Is it safe to fish here?",
                "latitude": 95.0,  # Invalid: > 90
                "longitude": 78.1348,
            }
        }

        resp = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp.status_code, 422, f"Expected HTTP 422 for invalid latitude, got: {resp.status_code}")
        print("✓ TEST 2 PASSED: Invalid latitude properly rejected with HTTP 422.")

    def test_03_invalid_longitude(self):
        """TEST 3: Out of range longitude (-180 to 180) returns HTTP 422 validation error."""
        payload = {
            "protocol_version": 1,
            "message_id": "test-lon-invalid",
            "message_type": "ORCA_QUERY",
            "timestamp": int(time.time()),
            "payload": {
                "query": "Is it safe to fish here?",
                "latitude": 8.7642,
                "longitude": 200.0,  # Invalid: > 180
            }
        }

        resp = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp.status_code, 422, f"Expected HTTP 422 for invalid longitude, got: {resp.status_code}")
        print("✓ TEST 3 PASSED: Invalid longitude properly rejected with HTTP 422.")

    def test_04_missing_message_id(self):
        """TEST 4: Missing or empty message_id returns HTTP 422 validation error."""
        payload = {
            "protocol_version": 1,
            "message_type": "ORCA_QUERY",
            "timestamp": int(time.time()),
            "payload": {
                "query": "Check weather offshore",
                "latitude": 8.7642,
                "longitude": 78.1348,
            }
        }

        resp = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp.status_code, 422, f"Expected HTTP 422 for missing message_id, got: {resp.status_code}")
        print("✓ TEST 4 PASSED: Missing message_id properly rejected with HTTP 422.")

    def test_05_duplicate_message_id_idempotency(self):
        """TEST 5: Duplicate message_id returns cached response with cached=True without re-running agents."""
        msg_id = f"test-idempotent-{uuid.uuid4().hex[:8]}"
        payload = {
            "protocol_version": 1,
            "message_id": msg_id,
            "message_type": "ORCA_QUERY",
            "timestamp": int(time.time()),
            "payload": {
                "query": "PFZ front detection test",
                "latitude": 8.7642,
                "longitude": 78.1348,
                "persona": "FISHERMAN",
            }
        }

        # First transmission
        resp1 = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertFalse(data1.get("cached", False))

        # Duplicate retransmission (simulating satellite retry)
        resp2 = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertEqual(data2.get("message_id"), msg_id)
        self.assertTrue(data2.get("cached", False), "Second transmission should have cached=True")
        self.assertEqual(data1.get("advisory"), data2.get("advisory"), "Cached advisory must match initial response")
        print("✓ TEST 5 PASSED: Idempotency verified - duplicate message served from cache.")

    def test_06_satellite_integration_disabled(self):
        """TEST 6: When SATELLITE_GATEWAY_ENABLED=false, returns clean HTTP 503 disabled response."""
        os.environ["SATELLITE_GATEWAY_ENABLED"] = "false"
        payload = {
            "protocol_version": 1,
            "message_id": "test-disabled-mode",
            "message_type": "ORCA_QUERY",
            "timestamp": int(time.time()),
            "payload": {
                "query": "Is it safe to fish?",
                "latitude": 8.7642,
                "longitude": 78.1348,
            }
        }

        resp = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp.status_code, 503, f"Expected HTTP 503 when disabled, got: {resp.status_code}")
        data = resp.json()
        self.assertEqual(data.get("status"), "DISABLED")
        self.assertIn("disabled", data.get("message", "").lower())
        print("✓ TEST 6 PASSED: Clean 503 Service Unavailable when satellite integration is disabled.")

    def test_07_gateway_unavailable_handling(self):
        """TEST 7: Handles unreachable outbound gateway URL gracefully without crashing."""
        os.environ["SATELLITE_GATEWAY_ENABLED"] = "true"
        os.environ["SATELLITE_GATEWAY_URL"] = "http://127.0.0.1:59999/nonexistent-gateway"
        os.environ["SATELLITE_MESSAGE_TIMEOUT"] = "1"

        comp_resp = SatelliteCompactResponse(
            protocol_version=1,
            message_id="test-unreachable-gw",
            status="SUCCESS",
            risk="SAFE",
            advisory="Test advisory",
            latitude=8.7642,
            longitude=78.1348,
            timestamp=int(time.time()),
        )

        # Test outbound send_compact_response with max_retries=1
        res = gateway_adapter.send_compact_response(comp_resp, max_retries=1)
        self.assertFalse(res.get("success"))
        self.assertEqual(res.get("reason"), "GATEWAY_TRANSMISSION_FAILED")
        self.assertEqual(res.get("message_id"), "test-unreachable-gw")
        print("✓ TEST 7 PASSED: Outbound gateway transmission error handled resiliently.")

    def test_08_normal_chat_endpoint_regression(self):
        """TEST 8: Confirms /api/chat still works without regression after server.py refactoring."""
        chat_req = {
            "query": "Is it safe to fish near Thoothukudi tomorrow?",
            "persona": "FISHERMAN",
            "lang": "en",
            "telemetry": {
                "latitude": 8.7642,
                "longitude": 78.1348,
                "speed_knots": 0.0,
                "heading_degrees": 120.0
            }
        }

        resp = self.client.post("/api/chat", json=chat_req)
        self.assertEqual(resp.status_code, 200, f"/api/chat returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        self.assertTrue(data.get("success") or data.get("status") == "success")
        self.assertIn("reply", data)
        self.assertIn("advisory", data)
        self.assertIn("satellite_provenance", data)
        print("✓ TEST 8 PASSED: /api/chat intact and regression-free.")

    def test_09_satellite_simulator_packet(self):
        """TEST 9: Simulates packet building and pipeline round-trip through simulator helper."""
        packet = build_simulated_satellite_packet(
            query="Check sea state and high waves",
            lat=8.7642,
            lon=78.1348,
            persona="FISHERMAN",
            message_id="sim-test-battery-09",
        )

        self.assertEqual(packet["protocol_version"], 1)
        self.assertEqual(packet["message_id"], "sim-test-battery-09")
        self.assertEqual(packet["payload"]["latitude"], 8.7642)

        resp = self.client.post("/api/satellite/message", json=packet)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["message_id"], "sim-test-battery-09")
        self.assertIn("advisory", data)
        print("✓ TEST 9 PASSED: Satellite simulator packet generated and processed cleanly.")

    def test_10_oversized_payload_rejected(self):
        """TEST 10: Rejects oversized payload (> SATELLITE_MAX_PAYLOAD_SIZE) with HTTP 413."""
        os.environ["SATELLITE_GATEWAY_ENABLED"] = "true"
        os.environ["SATELLITE_MAX_PAYLOAD_SIZE"] = "500"  # Small threshold for test

        huge_query = "A" * 600
        payload = {
            "protocol_version": 1,
            "message_id": "test-oversized",
            "message_type": "ORCA_QUERY",
            "timestamp": int(time.time()),
            "payload": {
                "query": huge_query,
                "latitude": 8.7642,
                "longitude": 78.1348,
            }
        }

        resp = self.client.post("/api/satellite/message", json=payload)
        self.assertEqual(resp.status_code, 413, f"Expected HTTP 413 for oversized payload, got: {resp.status_code}")
        data = resp.json()
        self.assertEqual(data.get("status"), "ERROR")
        self.assertIn("exceeds", data.get("message", "").lower())
        print("✓ TEST 10 PASSED: Oversized payload rejected safely with HTTP 413.")


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("RUNNING PROJECT ORCA SATELLITE GATEWAY TEST SUITE (10 TEST CASES)")
    print("=" * 75)
    unittest.main(verbosity=2)

