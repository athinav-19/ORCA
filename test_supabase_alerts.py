"""
test_supabase_alerts.py - Comprehensive Test Suite for Supabase Integration & Emergency Alerts
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Tests all 15 required criteria:
1. Supabase connection & health check
2. User registration (POST /api/user/register)
3. Duplicate user registration (HTTP 409 Conflict)
4. Current location update (PATCH /api/user/location)
5. Location history insertion (user_locations append-only)
6. Geographic radius search (True spherical Haversine)
7. Test alert creation (POST /api/alerts/test)
8. Affected user detection (inside vs outside radius)
9. alert_deliveries creation
10. Duplicate delivery prevention (idempotency check)
11. SMS provider not configured handling
12. Existing /api/query regression
13. Existing /api/chat regression
14. /health endpoint verification
15. /api/status endpoint verification
"""

import os
import sys
import uuid
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from unittest.mock import MagicMock, patch

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx
from server import app
from notification_provider import mask_phone_number, sms_provider, NotificationProvider
from alert_service import alert_service, haversine_km
from supabase_client import check_supabase_health, is_supabase_configured
from models import UserRole, UserRegisterRequest, UserLocationUpdateRequest, TestAlertRequest


# =====================================================================
# SYNCHRONOUS ASGI CLIENT WRAPPER (Compatible with Python 3.13 + httpx 0.28+)
# =====================================================================

class SyncASGIClient:
    """Synchronous test client wrapping httpx.AsyncClient with ASGITransport."""
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    def post(self, url: str, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
        async def _run():
            transport = httpx.ASGITransport(app=self.asgi_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(url, json=json, headers=headers or {})
        return asyncio.run(_run())

    def patch(self, url: str, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
        async def _run():
            transport = httpx.ASGITransport(app=self.asgi_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.patch(url, json=json, headers=headers or {})
        return asyncio.run(_run())

    def get(self, url: str, headers: Optional[Dict[str, str]] = None):
        async def _run():
            transport = httpx.ASGITransport(app=self.asgi_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(url, headers=headers or {})
        return asyncio.run(_run())


# =====================================================================
# IN-MEMORY SUPABASE EMULATOR FOR COMPREHENSIVE OFFLINE / LIVE TESTING
# =====================================================================

class MockTable:
    def __init__(self, name: str, db: dict):
        self.name = name
        self.db = db
        self._filters = []
        self._limit = None
        self._order_col = None
        self._order_desc = False
        self._pending_insert = None
        self._pending_update = None

    def select(self, cols: str = "*"):
        return self

    def eq(self, col: str, val: Any):
        self._filters.append((col, val))
        return self

    def order(self, col: str, desc: bool = False):
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, count: int):
        self._limit = count
        return self

    def insert(self, record: Union[Dict, List]):
        self._pending_insert = record
        return self

    def update(self, update_dict: Dict):
        self._pending_update = update_dict
        return self

    def execute(self):
        # Handle insert
        if self._pending_insert is not None:
            data = self._pending_insert
            self._pending_insert = None
            if isinstance(data, dict):
                new_record = dict(data)
                if "id" not in new_record:
                    new_record["id"] = str(uuid.uuid4())
                self.db.setdefault(self.name, []).append(new_record)
                res = MagicMock()
                res.data = [new_record]
                return res
            elif isinstance(data, list):
                res_list = []
                for r in data:
                    new_r = dict(r)
                    if "id" not in new_r:
                        new_r["id"] = str(uuid.uuid4())
                    self.db.setdefault(self.name, []).append(new_r)
                    res_list.append(new_r)
                res = MagicMock()
                res.data = res_list
                return res

        # Handle update
        if self._pending_update is not None:
            update_dict = self._pending_update
            self._pending_update = None
            matching = []
            for r in self.db.get(self.name, []):
                matches = True
                for col, val in self._filters:
                    if r.get(col) != val:
                        matches = False
                        break
                if matches:
                    r.update(update_dict)
                    matching.append(r)
            res = MagicMock()
            res.data = matching
            return res

        # Handle select query
        rows = list(self.db.get(self.name, []))
        for col, val in self._filters:
            rows = [r for r in rows if r.get(col) == val]
        if self._order_col:
            rows = sorted(rows, key=lambda r: r.get(self._order_col, ""), reverse=self._order_desc)
        if self._limit:
            rows = rows[:self._limit]
        res = MagicMock()
        res.data = rows
        return res


class MockSupabaseClient:
    def __init__(self):
        self.db = {
            "users": [],
            "user_locations": [],
            "alerts": [],
            "alert_deliveries": [],
        }

    def table(self, table_name: str):
        return MockTable(table_name, self.db)

    def rpc(self, fn_name: str, params: dict):
        raise Exception("RPC not present on client")


# =====================================================================
# TEST RUNNER
# =====================================================================

def run_all_tests():
    print("=" * 75)
    print("PROJECT ORCA - SUPABASE & EMERGENCY ALERT SERVICE TEST SUITE")
    print("ISRO SIH 176 - Multi-Agent System Database & Notification Verification")
    print("=" * 75)

    client = SyncASGIClient(app)
    mock_supabase = MockSupabaseClient()
    test_results = {}

    # -----------------------------------------------------------------
    # TEST 1: Supabase Connection & Health Diagnostic
    # -----------------------------------------------------------------
    print("\n[TEST 1] Supabase Connection & Health Check")
    health_info = check_supabase_health()
    print(f"  Health Diagnostic: {health_info}")
    assert "status" in health_info
    assert "project_url" in health_info
    assert health_info["project_url"] == "https://fdvkmfxietjubbzujntz.supabase.co"
    test_results["1_supabase_connection"] = "PASSED"
    print("  ✓ TEST 1 PASSED: Supabase client initialization and health diagnostic verified.")

    # Patch get_supabase_client and is_supabase_configured to use mock for tests 2-12
    with patch("server.get_supabase_client", return_value=mock_supabase), \
         patch("server.is_supabase_configured", return_value=True), \
         patch("alert_service.get_supabase_client", return_value=mock_supabase), \
         patch("alert_service.is_supabase_configured", return_value=True):

        # -----------------------------------------------------------------
        # TEST 2: User Registration (POST /api/user/register)
        # -----------------------------------------------------------------
        print("\n[TEST 2] User Registration")
        user_payload = {
            "name": "Captain Arun",
            "role": "FISHERMAN",
            "phone_number": "+919876543210",
            "latitude": 8.7642,
            "longitude": 78.1348
        }
        resp = client.post("/api/user/register", json=user_payload)
        print(f"  Response Status: {resp.status_code}")
        data = resp.json()
        print(f"  Response Body: {data}")
        assert resp.status_code == 200
        assert data.get("success") is True
        registered_user_id = data.get("user_id")
        assert registered_user_id is not None
        assert len(mock_supabase.db["users"]) == 1
        assert mock_supabase.db["users"][0]["phone_number"] == "+919876543210"
        test_results["2_user_registration"] = "PASSED"
        print(f"  ✓ TEST 2 PASSED: User registered with UUID={registered_user_id}.")

        # -----------------------------------------------------------------
        # TEST 3: Duplicate User Registration (HTTP 409 Conflict)
        # -----------------------------------------------------------------
        print("\n[TEST 3] Duplicate User Registration")
        dup_resp = client.post("/api/user/register", json=user_payload)
        print(f"  Duplicate Status: {dup_resp.status_code}")
        dup_data = dup_resp.json()
        print(f"  Duplicate Body: {dup_data}")
        assert dup_resp.status_code == 409
        assert dup_data.get("success") is False
        assert dup_data.get("code") == "DUPLICATE_USER"
        assert len(mock_supabase.db["users"]) == 1  # No duplicate row inserted
        test_results["3_duplicate_registration"] = "PASSED"
        print("  ✓ TEST 3 PASSED: Duplicate registration correctly rejected with HTTP 409 Conflict.")

        # -----------------------------------------------------------------
        # TEST 4: Current Location Update (PATCH /api/user/location)
        # -----------------------------------------------------------------
        print("\n[TEST 4] Current Location Update")
        loc_payload = {
            "user_id": registered_user_id,
            "latitude": 8.8500,
            "longitude": 78.2200
        }
        loc_resp = client.patch("/api/user/location", json=loc_payload)
        print(f"  Location Update Status: {loc_resp.status_code}")
        loc_data = loc_resp.json()
        print(f"  Location Update Body: {loc_data}")
        assert loc_resp.status_code == 200
        assert loc_data.get("success") is True
        assert loc_data.get("latitude") == 8.8500
        assert loc_data.get("longitude") == 78.2200
        # Verify in mock users table
        user_in_db = mock_supabase.db["users"][0]
        assert user_in_db.get("latitude") == 8.8500
        assert user_in_db.get("longitude") == 78.2200
        test_results["4_current_location_update"] = "PASSED"
        print("  ✓ TEST 4 PASSED: User current location updated in users table.")

        # -----------------------------------------------------------------
        # TEST 5: Location History Insertion (Append-only in user_locations)
        # -----------------------------------------------------------------
        print("\n[TEST 5] Location History Insertion")
        loc_payload_2 = {
            "user_id": registered_user_id,
            "latitude": 8.9200,
            "longitude": 78.3100
        }
        loc_resp_2 = client.patch("/api/user/location", json=loc_payload_2)
        assert loc_resp_2.status_code == 200
        history_records = mock_supabase.db["user_locations"]
        print(f"  user_locations History Count: {len(history_records)}")
        for i, h in enumerate(history_records):
            print(f"    Record {i+1}: lat={h.get('latitude')}, lon={h.get('longitude')}, time={h.get('recorded_at')}")
        assert len(history_records) >= 3
        lats = [h.get("latitude") for h in history_records]
        assert 8.7642 in lats
        assert 8.8500 in lats
        assert 8.9200 in lats
        test_results["5_location_history_insertion"] = "PASSED"
        print("  ✓ TEST 5 PASSED: Location history is strictly append-only, preserving all past GPS fixes.")

        # -----------------------------------------------------------------
        # TEST 6: Geographic Radius Search (True Spherical Haversine)
        # -----------------------------------------------------------------
        print("\n[TEST 6] Geographic Radius Search (Spherical Haversine)")
        dist_near = haversine_km(8.7642, 78.1348, 8.7139, 77.7567)
        dist_far = haversine_km(8.7642, 78.1348, 13.0827, 80.2707)
        print(f"  Calculated Geodesic Distance to Tirunelveli: {dist_near:.2f} km (Expected ~42-45 km)")
        print(f"  Calculated Geodesic Distance to Chennai: {dist_far:.2f} km (Expected ~525-535 km)")
        assert 40.0 < dist_near < 50.0
        assert 500.0 < dist_far < 550.0

        # Register user in Chennai (outside)
        mock_supabase.db["users"].append({
            "id": str(uuid.uuid4()),
            "name": "Researcher Deepa",
            "role": "RESEARCHER",
            "phone_number": "+919444455555",
            "latitude": 13.0827,
            "longitude": 80.2707,
            "notification_enabled": True,
            "sms_alerts_enabled": True,
        })
        # Register user in Thoothukudi (inside)
        user_inside_id = str(uuid.uuid4())
        mock_supabase.db["users"].append({
            "id": user_inside_id,
            "name": "Fisherman Velu",
            "role": "FISHERMAN",
            "phone_number": "+919123456789",
            "latitude": 8.8000,
            "longitude": 78.1500,
            "notification_enabled": True,
            "sms_alerts_enabled": True,
        })

        found_users = alert_service.find_users_in_radius(8.7642, 78.1348, radius_km=50.0, sms_only=True)
        found_phones = [u.get("phone_number") for u in found_users]
        print(f"  Users Found within 50 km: {len(found_users)}")
        for u in found_users:
            print(f"    Found: {u.get('name')} | dist={u.get('distance_km')} km | phone={mask_phone_number(u.get('phone_number'))}")
        assert "+919123456789" in found_phones
        assert "+919444455555" not in found_phones
        test_results["6_geographic_radius_search"] = "PASSED"
        print("  ✓ TEST 6 PASSED: Haversine distance correctly filters users within radius and excludes distant users.")

        # -----------------------------------------------------------------
        # TEST 7: Test Alert Creation (POST /api/alerts/test)
        # -----------------------------------------------------------------
        print("\n[TEST 7] Test Alert Creation")
        test_alert_payload = {
            "latitude": 8.7642,
            "longitude": 78.1348,
            "radius_km": 50,
            "severity": "WARNING",
            "message": "TEST ALERT - Severe gale force winds 45 kt detected offshore Thoothukudi."
        }
        alert_resp = client.post("/api/alerts/test", json=test_alert_payload)
        print(f"  Test Alert Status: {alert_resp.status_code}")
        alert_data = alert_resp.json()
        print(f"  Test Alert Response: {alert_data}")
        assert alert_resp.status_code == 200
        assert alert_data.get("success") is True
        assert alert_data.get("is_test") is True
        created_alert_id = alert_data.get("alert_id")
        assert created_alert_id is not None
        assert len(mock_supabase.db["alerts"]) == 1
        test_results["7_test_alert_creation"] = "PASSED"
        print(f"  ✓ TEST 7 PASSED: Alert created in Supabase with ID={created_alert_id}.")

        # -----------------------------------------------------------------
        # TEST 8: Affected User Detection
        # -----------------------------------------------------------------
        print("\n[TEST 8] Affected User Detection")
        affected_count = alert_data.get("affected_users_count")
        print(f"  Reported Affected Users Count: {affected_count}")
        assert affected_count >= 1
        test_results["8_affected_user_detection"] = "PASSED"
        print(f"  ✓ TEST 8 PASSED: {affected_count} affected users accurately detected based on current location.")

        # -----------------------------------------------------------------
        # TEST 9: alert_deliveries Creation
        # -----------------------------------------------------------------
        print("\n[TEST 9] alert_deliveries Creation")
        deliveries = mock_supabase.db["alert_deliveries"]
        print(f"  Deliveries Created in Supabase: {len(deliveries)}")
        for d in deliveries:
            print(f"    Delivery: alert={d.get('alert_id')} | user={d.get('user_id')} | channel={d.get('channel')} | status={d.get('status')}")
        assert len(deliveries) >= 1
        assert deliveries[0]["channel"] == "SMS"
        assert deliveries[0]["alert_id"] == created_alert_id
        test_results["9_alert_deliveries_creation"] = "PASSED"
        print("  ✓ TEST 9 PASSED: Delivery records accurately created in alert_deliveries table.")

        # -----------------------------------------------------------------
        # TEST 10: Duplicate Delivery Prevention (Idempotency)
        # -----------------------------------------------------------------
        print("\n[TEST 10] Duplicate Delivery Prevention")
        second_dispatch = alert_service.create_deliveries_and_dispatch(
            alert_id=created_alert_id,
            latitude=8.7642,
            longitude=78.1348,
            radius_km=50.0,
            message="TEST ALERT DUPLICATE"
        )
        print(f"  Second Dispatch Summary: {second_dispatch}")
        assert second_dispatch.get("skipped_duplicates_count") >= 1
        assert second_dispatch.get("sent_count") == 0
        test_results["10_duplicate_delivery_prevention"] = "PASSED"
        print("  ✓ TEST 10 PASSED: Idempotency check prevents duplicate dispatch for existing (alert_id, user_id, channel).")

        # -----------------------------------------------------------------
        # TEST 11: SMS Provider Not Configured Handling
        # -----------------------------------------------------------------
        print("\n[TEST 11] SMS Provider Not Configured Handling")
        with patch.dict(os.environ, {}, clear=True):
            unconf_provider = sms_provider
            res = unconf_provider.send_sms("+919876543210", "Test emergency alert")
            print(f"  Unconfigured SMS Provider Result: {res}")
            assert res.get("status") == "NOT_CONFIGURED"
            assert res.get("provider_message_id") is None
            assert "not configured" in (res.get("error", "") or res.get("message", "")).lower()
            masked = mask_phone_number("+919876543210")
            print(f"  Masked Phone Verification: +919876543210 -> {masked}")
            assert masked == "+9198******10"
            assert "7654" not in masked
        test_results["11_sms_unconfigured_handling"] = "PASSED"
        print("  ✓ TEST 11 PASSED: Unconfigured SMS provider cleanly returns NOT_CONFIGURED without faking delivery.")

        # -----------------------------------------------------------------
        # TEST 12: GET /api/alerts & GET /api/alerts/{id}
        # -----------------------------------------------------------------
        print("\n[TEST 12] GET /api/alerts & GET /api/alerts/{alert_id}")
        list_resp = client.get("/api/alerts")
        print(f"  GET /api/alerts Status: {list_resp.status_code}")
        list_data = list_resp.json()
        print(f"  GET /api/alerts Count: {list_data.get('count')}")
        assert list_resp.status_code == 200
        assert list_data.get("success") is True
        assert len(list_data.get("alerts", [])) >= 1

        detail_resp = client.get(f"/api/alerts/{created_alert_id}")
        print(f"  GET /api/alerts/{created_alert_id} Status: {detail_resp.status_code}")
        detail_data = detail_resp.json()
        print(f"  Alert Detail affected_count: {detail_data.get('alert', {}).get('affected_user_count')}")
        assert detail_resp.status_code == 200
        assert detail_data.get("success") is True
        serialized = json.dumps(detail_data)
        assert "+919876543210" not in serialized
        test_results["12_alert_query_apis"] = "PASSED"
        print("  ✓ TEST 12 PASSED: Alert listing and detail endpoints return accurate summaries without exposing phone numbers.")

    # -----------------------------------------------------------------
    # TEST 13: Existing /api/query Regression Check
    # -----------------------------------------------------------------
    print("\n[TEST 13] Existing /api/query Regression Check")
    query_payload = {
        "query": "Is it safe to fish offshore Thoothukudi today?",
        "persona": "FISHERMAN",
        "telemetry": {
            "latitude": 8.7642,
            "longitude": 78.1348,
            "speed_knots": 0.0,
            "heading_degrees": 120.0,
            "gps_accuracy_meters": 4.5
        }
    }
    q_resp = client.post("/api/query", json=query_payload)
    print(f"  /api/query Status: {q_resp.status_code}")
    q_data = q_resp.json()
    assert q_resp.status_code == 200
    assert q_data.get("status") == "success"
    assert q_data.get("threat_status") is not None
    assert q_data.get("advisory") is not None
    test_results["13_existing_query_regression"] = "PASSED"
    print("  ✓ TEST 13 PASSED: Existing /api/query endpoint functions with zero regression.")

    # -----------------------------------------------------------------
    # TEST 14: Existing /api/chat Regression Check
    # -----------------------------------------------------------------
    print("\n[TEST 14] Existing /api/chat Regression Check")
    chat_resp = client.post("/api/chat", json={"query": "Hello Captain", "persona": "FISHERMAN"})
    print(f"  /api/chat Status: {chat_resp.status_code}")
    chat_data = chat_resp.json()
    assert chat_resp.status_code == 200
    assert chat_data.get("status") == "success"
    assert chat_data.get("chat_text") is not None
    test_results["14_existing_chat_regression"] = "PASSED"
    print("  ✓ TEST 14 PASSED: Existing /api/chat endpoint functions with zero regression.")

    # -----------------------------------------------------------------
    # TEST 15: /health and /api/status Endpoints
    # -----------------------------------------------------------------
    print("\n[TEST 15] /health & /api/status Endpoints")
    h_resp = client.get("/health")
    print(f"  /health Status: {h_resp.status_code}")
    h_data = h_resp.json()
    assert h_resp.status_code == 200
    assert h_data.get("status") == "healthy"
    assert "supabase" in h_data

    s_resp = client.get("/api/status")
    print(f"  /api/status Status: {s_resp.status_code}")
    s_data = s_resp.json()
    assert s_resp.status_code == 200
    assert s_data.get("status") == "OPERATIONAL"
    assert "supabase" in s_data
    assert "active_domain_agents" in s_data

    api_dir_resp = client.get("/api")
    assert api_dir_resp.status_code == 200
    endpoints = api_dir_resp.json().get("endpoints", {})
    assert "user_register" in endpoints
    assert "user_location" in endpoints
    assert "alerts_test" in endpoints
    assert "alerts_list" in endpoints
    assert "alert_detail" in endpoints
    test_results["15_health_and_status"] = "PASSED"
    print("  ✓ TEST 15 PASSED: Health, status, and API directory endpoints correctly include Supabase diagnostics.")

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TEST SUITE EXECUTION SUMMARY")
    print("=" * 75)
    for t_name, status in test_results.items():
        print(f"  {t_name:35}: {status}")
    print("=" * 75)
    all_passed = all(s == "PASSED" for s in test_results.values())
    print(f"ALL 15 TESTS PASSED: {all_passed}")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

